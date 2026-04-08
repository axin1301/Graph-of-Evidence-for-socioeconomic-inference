# evidence_aggregator.py

def aggregate_single_group(evidences, target_field):
    positive_score = 0.0
    negative_score = 0.0
    neutral_count = 0

    for e in evidences:
        implication = e["implication"].get(target_field, 0)
        confidence = e["confidence"]

        if implication > 0:
            positive_score += implication * confidence
        elif implication < 0:
            negative_score += abs(implication) * confidence
        else:
            neutral_count += 1

    return {
        "positive_score": round(positive_score, 3),
        "negative_score": round(negative_score, 3),
        "net_score": round(positive_score - negative_score, 3),
        "neutral_count": neutral_count,
        "evidence_count": len(evidences)
    }


def aggregate_evidence(evidences, target_field, use_balanced=True):
    if not use_balanced:
        overall = aggregate_single_group(evidences, target_field)
        return {
            "overall": overall
        }

    sat = [e for e in evidences if e["modality"] == "satellite"]
    st = [e for e in evidences if e["modality"] == "street"]

    sat_stats = aggregate_single_group(sat, target_field)
    st_stats = aggregate_single_group(st, target_field)

    balanced_net = 0.5 * sat_stats["net_score"] + 0.5 * st_stats["net_score"]

    return {
        "satellite": sat_stats,
        "street": st_stats,
        "balanced": {
            "net_score": round(balanced_net, 3)
        }
    }

