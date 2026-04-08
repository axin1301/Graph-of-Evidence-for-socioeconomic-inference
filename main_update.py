from agents.satellite_agent import satellite_agent
from agents.street_agent import street_agent
from agents.claim_agent import gdp_claim_agent
from agents.verification_agent import verification_agent
from agents.refinement_agent import refinement_agent
from agents.final_report_agent import final_report_agent
from core.io_utils import save_json
from core.task_parser import parse_task_from_query
from agents.reflection_agent import reflection_agent
from core.task_guidance import canonicalize_task_guidance_key
import os
import json
from core.schemas import Claim

from graph.goe_builder import build_base_goe
from graph.goe_augmentor import augment_goe_with_candidates
from graph.edge_potential import infer_edge_potentials


MAX_STREET_IMAGES = int(os.getenv("MAX_STREET_IMAGES", "10"))


def _sample_street_images(street_images, max_images):
    if max_images is None or max_images <= 0 or len(street_images) <= max_images:
        return street_images

    if max_images == 1:
        return [street_images[len(street_images) // 2]]

    selected = []
    last_idx = len(street_images) - 1
    for i in range(max_images):
        idx = round(i * last_idx / (max_images - 1))
        selected.append(street_images[idx])
    return selected


def run_single_case(
    original_task_name,
    case_id,
    sat_image,
    street_images,
    query,
    use_mock=True,
    output_dir="outputs_update",
    use_verification=True,
    use_refinement=True,
    setting_name="default",
    use_contradiction_handling=True,
    use_reflection=True,
    use_balanced_aggregation=True,
):
    case_output_dir = f"{output_dir}/{original_task_name}/{case_id}"
    if os.path.exists(f"{case_output_dir}/goe.json"):
        return 0

    os.makedirs(case_output_dir, exist_ok=True)

    print("task parser running...")
    task_spec = parse_task_from_query(query, use_mock=use_mock)
    save_json(task_spec, f"{case_output_dir}/task_spec.json")

    street_images = _sample_street_images(street_images, MAX_STREET_IMAGES)

    print("satellite agent running...")
    sat_evidences = satellite_agent(sat_image, query, task_spec, use_mock=use_mock)
    save_json([e.to_dict() for e in sat_evidences], f"{case_output_dir}/sat_evidences.json")

    print("street agent running...")
    street_evidences = street_agent(street_images, query, task_spec, use_mock=use_mock)
    save_json([e.to_dict() for e in street_evidences], f"{case_output_dir}/street_evidences.json")

    all_evidences = sat_evidences + street_evidences
    all_evidences_dict = [e.to_dict() for e in all_evidences]

    print("claim agent running...")
    claim = gdp_claim_agent(all_evidences, query, task_spec, use_mock=use_mock)
    save_json(claim.to_dict(), f"{case_output_dir}/initial_claim_cached.json")

    reasoning_trace = {
        "all_evidences": all_evidences_dict,
        "initial_claim": None,
        "initial_verification": None,
        "refinement_used": False,
        "refined_claim": None,
        "post_refinement_verification": None,
        "reflection_triggered": False,
        "reflected_claim": None,
        "final_verification": None,
    }
    reasoning_trace["initial_claim"] = claim.to_dict()

    final_claim = claim
    initial_verification_result = None
    final_verification_result = None

    if use_verification:
        print("verification agent running...")
        initial_verification_result = verification_agent(claim, all_evidences, task_spec)
        reasoning_trace["initial_verification"] = initial_verification_result
        final_verification_result = initial_verification_result

        if initial_verification_result["status"] == "FAIL" and use_refinement:
            reasoning_trace["refinement_used"] = True
            print("refinement agent running...")
            refined_claim, contradict_eids = refinement_agent(
                claim,
                all_evidences,
                initial_verification_result,
                query,
                task_spec,
                use_mock=use_mock,
                use_contradiction_handling=use_contradiction_handling,
            )

            if refined_claim is not None:
                reasoning_trace["refined_claim"] = refined_claim.to_dict()
                final_claim = refined_claim

            print("verification agent running...")
            final_verification_result = verification_agent(final_claim, all_evidences, task_spec)
            reasoning_trace["post_refinement_verification"] = final_verification_result

            should_reflect = final_verification_result["status"] == "FAIL" and use_reflection
            if should_reflect:
                issues = set(final_verification_result.get("issues", []))
                support_summary = final_verification_result.get("support_summary", {}) or {}
                strong_conflict = support_summary.get("negative_conflict_score", 0.0) >= 0.95
                only_soft_contradiction = issues.issubset({"UNADDRESSED_CONTRADICTION"})
                if only_soft_contradiction and not strong_conflict:
                    should_reflect = False
                if canonicalize_task_guidance_key(task_spec.get("target_field")) == "population":
                    if support_summary.get("negative_conflict_score", 0.0) < 1.10:
                        should_reflect = False

            if should_reflect:
                reasoning_trace["reflection_triggered"] = True
                print("reflection agent running...")
                reflected_claim = reflection_agent(
                    final_claim,
                    all_evidences,
                    final_verification_result,
                    task_spec,
                    query,
                    use_mock=use_mock,
                )

                if reflected_claim is not None:
                    reasoning_trace["reflected_claim"] = reflected_claim.to_dict()
                    final_claim = reflected_claim

                print("verification agent running...")
                final_verification_result = verification_agent(final_claim, all_evidences, task_spec)
                reasoning_trace["final_verification"] = final_verification_result

    if reasoning_trace["final_verification"] is None:
        reasoning_trace["final_verification"] = final_verification_result

    print("goe builder running...")
    base_goe = build_base_goe(
        all_evidences,
        final_claim,
        initial_verification=initial_verification_result,
        final_verification=final_verification_result,
        case_id=case_id,
        setting_name=setting_name,
        task_spec=task_spec,
        reasoning_trace=reasoning_trace,
        use_balanced_aggregation=use_balanced_aggregation,
    )

    print("goe augmentor running...")
    goe_aug = augment_goe_with_candidates(
        base_goe,
        add_evidence_claim_candidates=True,
        add_evidence_evidence_candidates=True,
        add_factor_nodes=True,
        add_evidence_factor_candidates=True,
        add_factor_claim_edges=True,
    )

    from graph.claim_solver import solve_goe_claims_v3

    print("edge potential running...")
    edge_potentials = infer_edge_potentials(
        goe_aug,
        task_spec,
        evidence_factor_topk=3,
        evidence_factor_min_score=0.12,
        factor_prior_scale=0.7,
    )

    print("claim solver running...")
    solver_result = solve_goe_claims_v3(
        goe_aug,
        edge_potentials,
        task_spec,
        factor_prior_scale=0.7,
        redundancy_penalty_scale=0.05,
        verification_bonus_scale=0.5,
        estimate_temperature=1.1,
    )

    goe_aug["edge_potentials"] = edge_potentials
    goe_aug["solver_result"] = solver_result

    print("final report agent running...")
    final_report = final_report_agent(goe_aug, query, task_spec, use_mock=use_mock)
    final_report["case_id"] = case_id
    final_report["setting"] = setting_name

    save_json(base_goe, f"{case_output_dir}/base_goe.json")
    save_json(goe_aug, f"{case_output_dir}/augmented_goe.json")
    save_json(edge_potentials, f"{case_output_dir}/edge_potentials.json")
    save_json(solver_result, f"{case_output_dir}/solver_result.json")
    save_json(reasoning_trace, f"{case_output_dir}/reasoning_trace.json")
    save_json(final_report, f"{case_output_dir}/final_report.json")
    save_json(initial_verification_result, f"{case_output_dir}/initial_verification.json")
    save_json(final_verification_result, f"{case_output_dir}/final_verification.json")

    return {
        "case_id": case_id,
        "task_spec": task_spec,
        "base_goe": base_goe,
        "augmented_goe": goe_aug,
        "edge_potentials": edge_potentials,
        "solver_result": solver_result,
        "final_report": final_report,
        "initial_verification": initial_verification_result,
        "final_verification": final_verification_result,
        "reasoning_trace": reasoning_trace,
    }



