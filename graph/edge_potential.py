from core.schemas_v2 import EdgePotential
from graph.factor_library import FACTOR_KEYWORDS, FACTOR_TASK_PRIOR

def _structured_factor_match(evidence, factor_name):
    score = 0.0

    layout = evidence.get("spatial_layout", {}) or {}
    key_elements = evidence.get("key_elements", []) or []

    # flatten structured text
    structured_tokens = []

    for _, v in layout.items():
        if isinstance(v, str):
            structured_tokens.extend(v.lower().split())

    for item in key_elements:
        if isinstance(item, dict):
            t = item.get("type", "")
            d = item.get("description", "")
            structured_tokens.extend(str(t).lower().split())
            structured_tokens.extend(str(d).lower().split())

    structured_tokens = set(structured_tokens)

    kws = FACTOR_KEYWORDS.get(factor_name, [])
    if not kws:
        return 0.0

    best = 0.0
    for kw in kws:
        kw_tokens = _tokenize(kw)
        score_kw = _jaccard(structured_tokens, kw_tokens)
        if score_kw > best:
            best = score_kw

    return best


def _coverage_weight(coverage):
    if coverage is None:
        return 0.55
    coverage = str(coverage).lower()
    if coverage == "broad":
        return 0.75
    if coverage == "medium":
        return 0.65
    if coverage in ("corridor", "broader_street_segment"):
        return 0.60
    if coverage == "local":
        return 0.50
    if coverage == "unclear":
        return 0.45
    return 0.55

def _safe_get_implication(evidence, target_field):
    imp = evidence.get("implication", {})
    try:
        return float(imp.get(target_field, 0.0))
    except Exception:
        return 0.0


def _tokenize(text):
    return set(str(text).lower().replace(",", " ").replace(".", " ").split())


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _source_prior(source_name):
    if source_name is None:
        return 0.55
    return 0.60


def _modality_prior(modality, factor_name=None):
    # light, task-agnostic prior
    if factor_name is None:
        return 0.5
    if factor_name in ("density_intensity", "connectivity_accessibility", "industrial_activity"):
        return 0.65 if modality == "satellite" else 0.45
    if factor_name in ("residential_quality", "built_environment_order", "commercial_activity", "green_amenity"):
        return 0.65 if modality == "street" else 0.45
    return 0.5


def _text_factor_match(observation, factor_name):
    obs_tokens = _tokenize(observation)
    kws = FACTOR_KEYWORDS.get(factor_name, [])
    if not kws:
        return 0.0

    best = 0.0
    for kw in kws:
        score = _jaccard(obs_tokens, _tokenize(kw))
        if score > best:
            best = score
    return best


def infer_edge_potentials(
    goe_aug,
    task_spec,
    evidence_factor_topk=2,
    evidence_factor_min_score=0.18,
    factor_prior_scale=0.7
):
    potentials = []
    evidence_factor_candidates = []

    target_field = task_spec["target_field"]

    evidences = {e["id"]: e for e in goe_aug.get("evidences", [])}
    claims = {c["id"]: c for c in goe_aug.get("claims", [])}
    factors = {f["id"]: f for f in goe_aug.get("factors", [])}

    potentials = []

    for rel in goe_aug.get("relations", []):
        rtype = rel["relation_type"]
        from_id = rel["from_id"]
        to_id = rel["to_id"]

        # -----------------------------
        # evidence -> claim
        # -----------------------------
        if from_id in evidences and to_id in claims and rtype in ("support", "contradict", "candidate_evidence_claim"):
            e = evidences[from_id]

            implication = _safe_get_implication(e, target_field)
            conf = float(e.get("confidence", 0.5))
            src_prior = _source_prior(e.get("source_name"))

            support_strength = max(0.0, implication) * conf
            conflict_strength = max(0.0, -implication) * conf

            potential_score = support_strength - conflict_strength

            if rtype == "support":
                potential_score += 0.12
            elif rtype == "contradict":
                potential_score -= 0.12

            potential_score = 0.85 * potential_score + 0.15 * src_prior

            potentials.append(EdgePotential(
                edge_id=rel["id"],
                support_strength=round(support_strength, 4),
                conflict_strength=round(conflict_strength, 4),
                redundancy_strength=0.0,
                source_reliability=round(src_prior, 4),
                implication_alignment=round(implication, 4),
                potential_score=round(potential_score, 4),
                rationale={
                    "edge_family": "evidence_claim",
                    "evidence_confidence": conf,
                    "observed_relation_type": rtype
                }
            ).to_dict())

        # -----------------------------
        # evidence -> factor
        # -----------------------------
        elif from_id in evidences and to_id in factors and rtype in ("candidate_evidence_factor", "evidence_factor"):
            e = evidences[from_id]
            f = factors[to_id]

            factor_name = f["factor_name"]
            obs = e.get("observation", "")
            conf = float(e.get("confidence", 0.5))
            implication = _safe_get_implication(e, target_field)

            text_match = _text_factor_match(obs, factor_name)
            structured_match = _structured_factor_match(e, factor_name)
            mod_prior = _modality_prior(e.get("modality"), factor_name)
            src_prior = _source_prior(e.get("source_name"))
            coverage_weight = _coverage_weight(e.get("coverage"))

            support_strength = (
                0.30 * text_match +
                0.30 * structured_match +
                0.15 * implication +
                0.15 * mod_prior +
                0.05 * src_prior +
                0.05 * coverage_weight
            ) * conf

            support_strength = max(0.0, min(1.0, support_strength))

            evidence_factor_candidates.append({
                "edge_id": rel["id"],
                "from_id": from_id,
                "to_id": to_id,
                "support_strength": round(support_strength, 4),
                "conflict_strength": 0.0,
                "redundancy_strength": 0.0,
                "source_reliability": round(src_prior, 4),
                "implication_alignment": round(implication, 4),
                "potential_score": round(support_strength, 4),
                "rationale": {
                    "edge_family": "evidence_factor",
                    "text_match": round(text_match, 4),
                    "modality_prior": round(mod_prior, 4),
                    "evidence_confidence": conf,
                    "structured_match": round(structured_match, 4),
                    "coverage_weight": round(coverage_weight, 4),
                }
            })

        # -----------------------------
        # factor -> claim
        # -----------------------------
        elif from_id in factors and to_id in claims and rtype == "factor_claim":
            f = factors[from_id]
            factor_name = f["factor_name"]
            prior = FACTOR_TASK_PRIOR.get(factor_name, {}).get(target_field, 0.0) * factor_prior_scale
            support_strength = max(0.0, prior)
            conflict_strength = max(0.0, -prior)

            potentials.append(EdgePotential(
                edge_id=rel["id"],
                support_strength=round(support_strength, 4),
                conflict_strength=round(conflict_strength, 4),
                redundancy_strength=0.0,
                source_reliability=0.0,
                implication_alignment=round(prior, 4),
                potential_score=round(prior, 4),
                rationale={
                    "edge_family": "factor_claim",
                    "factor_name": factor_name
                }
            ).to_dict())

        # -----------------------------
        # evidence <-> evidence redundancy
        # -----------------------------
        elif from_id in evidences and to_id in evidences and rtype == "candidate_redundancy":
            ei = evidences[from_id]
            ej = evidences[to_id]

            sim = _jaccard(_tokenize(ei.get("observation", "")), _tokenize(ej.get("observation", "")))
            same_modality = 1.0 if ei.get("modality") == ej.get("modality") else 0.0
            same_source = 1.0 if ei.get("source_name") == ej.get("source_name") else 0.0

            imp_i = _safe_get_implication(ei, target_field)
            imp_j = _safe_get_implication(ej, target_field)
            implication_similarity = 1.0 - min(1.0, abs(imp_i - imp_j) / 2.0)

            redundancy = (
                0.40 * sim +
                0.25 * same_modality +
                0.20 * same_source +
                0.15 * implication_similarity
            )
            redundancy = max(0.0, min(1.0, redundancy))

            potentials.append(EdgePotential(
                edge_id=rel["id"],
                support_strength=0.0,
                conflict_strength=0.0,
                redundancy_strength=round(redundancy, 4),
                source_reliability=0.0,
                implication_alignment=0.0,
                potential_score=round(-redundancy, 4),
                rationale={
                    "edge_family": "redundancy",
                    "lexical_jaccard": round(sim, 4),
                    "same_modality": same_modality,
                    "same_source": same_source,
                    "implication_similarity": round(implication_similarity, 4)
                }
            ).to_dict())

    # keep only top-k evidence->factor edges per evidence
    ef_by_evidence = {}
    for item in evidence_factor_candidates:
        ef_by_evidence.setdefault(item["from_id"], []).append(item)

    for eid, items in ef_by_evidence.items():
        items = sorted(items, key=lambda x: x["support_strength"], reverse=True)

        kept = 0
        for item in items:
            if item["support_strength"] < evidence_factor_min_score:
                continue
            if kept >= evidence_factor_topk:
                break

            potentials.append(EdgePotential(
                edge_id=item["edge_id"],
                support_strength=item["support_strength"],
                conflict_strength=item["conflict_strength"],
                redundancy_strength=item["redundancy_strength"],
                source_reliability=item["source_reliability"],
                implication_alignment=item["implication_alignment"],
                potential_score=item["potential_score"],
                rationale=item["rationale"]
            ).to_dict())
            kept += 1

    return {
        "method": "hierarchical_training_free_edge_potential_v3",
        "target_field": target_field,
        "potentials": potentials
    }



