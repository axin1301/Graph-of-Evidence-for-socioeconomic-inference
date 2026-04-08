from copy import deepcopy
from graph.factor_library import FACTOR_LIBRARY, FACTOR_TASK_PRIOR


def relation_exists(relations, from_id, to_id, relation_type):
    for r in relations:
        if (
            r["from_id"] == from_id and
            r["to_id"] == to_id and
            r["relation_type"] == relation_type
        ):
            return True
    return False


def augment_goe_with_candidates(
    goe,
    add_evidence_claim_candidates=True,
    add_evidence_evidence_candidates=True,
    add_factor_nodes=True,
    add_evidence_factor_candidates=True,
    add_factor_claim_edges=True
):
    goe_aug = deepcopy(goe)

    relations = goe_aug.get("relations", [])
    evidences = goe_aug.get("evidences", [])
    claims = goe_aug.get("claims", [])

    if "factors" not in goe_aug:
        goe_aug["factors"] = []

    # 1) add factor nodes
    if add_factor_nodes and len(goe_aug["factors"]) == 0:
        target_field = goe_aug["meta"].get("target_field")
        for factor in FACTOR_LIBRARY:
            task_rel = {}
            if factor["factor_name"] in FACTOR_TASK_PRIOR:
                task_rel[target_field] = FACTOR_TASK_PRIOR[factor["factor_name"]].get(target_field, 0.0)

            goe_aug["factors"].append({
                "id": factor["id"],
                "factor_name": factor["factor_name"],
                "description": factor["description"],
                "task_relevance": task_rel
            })

    factors = goe_aug.get("factors", [])

    # 2) evidence -> claim candidate edges
    if add_evidence_claim_candidates:
        for e in evidences:
            for c in claims:
                has_support = relation_exists(relations, e["id"], c["id"], "support")
                has_contradict = relation_exists(relations, e["id"], c["id"], "contradict")
                has_candidate = relation_exists(relations, e["id"], c["id"], "candidate_evidence_claim")

                if not has_support and not has_contradict and not has_candidate:
                    relations.append({
                        "id": f"cand::{e['id']}::eclaim::{c['id']}",
                        "from_id": e["id"],
                        "to_id": c["id"],
                        "relation_type": "candidate_evidence_claim",
                        "observed": False,
                        "weight": None,
                        "metadata": {}
                    })

    # 3) evidence <-> evidence redundancy candidates
    if add_evidence_evidence_candidates:
        n = len(evidences)
        for i in range(n):
            for j in range(i + 1, n):
                ei = evidences[i]
                ej = evidences[j]

                if not relation_exists(relations, ei["id"], ej["id"], "candidate_redundancy"):
                    relations.append({
                        "id": f"cand::{ei['id']}::redundant::{ej['id']}",
                        "from_id": ei["id"],
                        "to_id": ej["id"],
                        "relation_type": "candidate_redundancy",
                        "observed": False,
                        "weight": None,
                        "metadata": {}
                    })

    # 4) evidence -> factor candidate edges
    if add_evidence_factor_candidates:
        for e in evidences:
            for f in factors:
                if not relation_exists(relations, e["id"], f["id"], "candidate_evidence_factor"):
                    relations.append({
                        "id": f"cand::{e['id']}::efactor::{f['id']}",
                        "from_id": e["id"],
                        "to_id": f["id"],
                        "relation_type":"evidence_factor",
                        "observed": "True",
                        "weight": None,
                        "metadata": {}
                    })

    # 5) factor -> claim edges
    if add_factor_claim_edges:
        for f in factors:
            for c in claims:
                if not relation_exists(relations, f["id"], c["id"], "factor_claim"):
                    relations.append({
                        "id": f"edge::{f['id']}::factor_claim::{c['id']}",
                        "from_id": f["id"],
                        "to_id": c["id"],
                        "relation_type": "factor_claim",
                        "observed": True,
                        "weight": None,
                        "metadata": {}
                    })

    goe_aug["relations"] = relations
    goe_aug["meta"]["graph_version"] = "goe_augmented_v3"
    return goe_aug



