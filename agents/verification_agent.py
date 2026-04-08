from core.task_guidance import canonicalize_task_guidance_key


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def verification_agent(claim, evidences, task_spec):
    """
    Check:
    1. whether there is enough support evidence
    2. whether support covers both satellite and street modalities
    3. whether contradictions are left unaddressed
    4. whether the estimate is roughly aligned with evidence strength
    """

    target_field = task_spec["target_field"]

    support_set = set(claim.support_eids)
    contradict_set = set(getattr(claim, "contradict_eids", []))

    satellite_support_count = 0
    street_support_count = 0

    positive_support_score = 0.0
    negative_conflict_score = 0.0

    street_positive_count = 0
    street_negative_count = 0
    street_strong_negative_count = 0
    street_neutral_or_weak_count = 0
    street_positive_score = 0.0
    street_negative_score = 0.0
    street_strong_negative_score = 0.0

    eid_to_evidence = {e.eid: e for e in evidences}

    for eid in claim.support_eids:
        if eid not in eid_to_evidence:
            continue

        e = eid_to_evidence[eid]

        if e.modality == "satellite":
            satellite_support_count += 1
        elif e.modality == "street":
            street_support_count += 1

        implication = e.implication.get(target_field, 0)
        try:
            implication_value = float(implication)
        except Exception:
            implication_value = 0.0

        if implication_value >= 0:
            positive_support_score += implication_value * e.confidence

        if e.modality == "street":
            informativeness = _safe_float(getattr(e, "informativeness", 0.5), 0.5)
            scene_type = str(getattr(e, "scene_type", "") or "").lower()
            strong_street_signal = (
                scene_type != "weakly_informative"
                and informativeness >= 0.45
                and e.confidence >= 0.45
            )
            if implication_value > 0.05:
                street_positive_count += 1
                street_positive_score += implication_value * e.confidence
            elif implication_value < -0.05:
                street_negative_count += 1
                street_negative_score += abs(implication_value) * e.confidence
                if implication_value <= -0.20 and strong_street_signal:
                    street_strong_negative_count += 1
                    street_strong_negative_score += abs(implication_value) * e.confidence
            else:
                street_neutral_or_weak_count += 1

    unused_conflicts = []
    addressed_conflicts = []

    for e in evidences:
        implication = e.implication.get(target_field, 0)
        try:
            implication_value = float(implication)
        except Exception:
            implication_value = 0.0

        if implication_value < -0.05:
            if e.eid in contradict_set:
                addressed_conflicts.append(e.eid)
                negative_conflict_score += abs(implication_value) * e.confidence
            elif e.eid not in support_set:
                unused_conflicts.append(e.eid)
                negative_conflict_score += abs(implication_value) * e.confidence

    target_key = canonicalize_task_guidance_key(target_field)
    issues = []

    if len(claim.support_eids) < 2:
        issues.append("INSUFFICIENT_SUPPORT_COUNT")

    if satellite_support_count < 1:
        issues.append("MISSING_SATELLITE_SUPPORT")

    if street_support_count < 1:
        issues.append("MISSING_STREET_SUPPORT")

    if len(unused_conflicts) > 0:
        issues.append("UNADDRESSED_CONTRADICTION")

    if positive_support_score < 1.2 and claim.estimate > 6.0:
        issues.append("ESTIMATE_TOO_HIGH_FOR_SUPPORT")

    if negative_conflict_score >= 0.7 and claim.estimate > 6.0:
        issues.append("ESTIMATE_IGNORES_CONTRADICTION")
    if target_key == "gdp":
        if street_strong_negative_count >= max(3, street_positive_count + 2) and claim.estimate >= 5.3:
            issues.append("HIGH_ESTIMATE_CONFLICTS_WITH_STREET_SIGNAL")

        if street_strong_negative_score > street_positive_score + 0.20 and claim.estimate >= 5.0:
            issues.append("STREET_EVIDENCE_TRENDS_LOWER_THAN_CLAIM")

        if street_positive_count <= 1 and street_strong_negative_count >= 4 and claim.estimate >= 5.8:
            issues.append("INSUFFICIENT_POSITIVE_STREET_SUPPORT_FOR_HIGH_GDP")
    elif target_key == "population":
        if street_strong_negative_count >= max(3, street_positive_count + 1) and claim.estimate >= 5.5:
            issues.append("HIGH_POPULATION_CLAIM_CONFLICTS_WITH_STREET_SIGNAL")

        if street_strong_negative_score > street_positive_score + 0.16 and claim.estimate >= 5.2:
            issues.append("STREET_EVIDENCE_TRENDS_LOWER_THAN_POPULATION_CLAIM")
    elif target_key == "carbon":
        if street_positive_count == 0 and claim.estimate >= 6.0:
            issues.append("INSUFFICIENT_ACTIVITY_SUPPORT_FOR_HIGH_CARBON")

        if street_strong_negative_score > street_positive_score + 0.16 and claim.estimate >= 5.4:
            issues.append("STREET_EVIDENCE_TRENDS_LOWER_THAN_CARBON_CLAIM")
    elif target_key == "houseprice":
        if street_strong_negative_count >= max(2, street_positive_count + 1) and claim.estimate >= 5.4:
            issues.append("HIGH_HOUSEPRICE_CLAIM_CONFLICTS_WITH_STREET_SIGNAL")

        if street_positive_count == 0 and claim.estimate >= 6.0:
            issues.append("INSUFFICIENT_POSITIVE_STREET_SUPPORT_FOR_HIGH_HOUSEPRICE")
    elif target_key == "bachelorratio":
        if street_strong_negative_score > street_positive_score + 0.16 and claim.estimate >= 5.4:
            issues.append("HIGH_BACHELOR_RATIO_CLAIM_CONFLICTS_WITH_STREET_SIGNAL")

        if street_positive_count == 0 and claim.estimate >= 5.8:
            issues.append("INSUFFICIENT_POSITIVE_STREET_SUPPORT_FOR_HIGH_BACHELOR_RATIO")
    elif target_key == "violentcrime":
        if street_strong_negative_count >= max(3, street_positive_count + 1) and claim.estimate >= 5.4:
            issues.append("HIGH_CRIME_CLAIM_CONFLICTS_WITH_ORDERLY_STREET_SIGNAL")

        if street_positive_count == 0 and claim.estimate >= 5.8:
            issues.append("INSUFFICIENT_POSITIVE_STREET_SUPPORT_FOR_HIGH_CRIME")
    elif target_key == "buildheight":
        if street_strong_negative_count >= max(2, street_positive_count + 1) and claim.estimate >= 5.8:
            issues.append("HIGH_BUILDHEIGHT_CLAIM_CONFLICTS_WITH_LOW_RISE_STREET_SIGNAL")

        # Build height should rely primarily on large-scale morphology. Do not
        # fail a high-rise claim merely because street views are limited or not
        # strongly positive, unless satellite support is also weak.
        if street_positive_count == 0 and satellite_support_count == 0 and claim.estimate >= 6.4:
            issues.append("INSUFFICIENT_POSITIVE_STREET_SUPPORT_FOR_HIGH_BUILDHEIGHT")

    status = "PASS" if len(issues) == 0 else "FAIL"

    result = {
        "task": task_spec["task_name"],
        "target_field": target_field,
        "status": status,
        "issues": issues,
        "support_summary": {
            "support_count": len(claim.support_eids),
            "satellite_support_count": satellite_support_count,
            "street_support_count": street_support_count,
            "positive_support_score": round(positive_support_score, 3),
            "negative_conflict_score": round(negative_conflict_score, 3),
            "street_positive_count": street_positive_count,
            "street_negative_count": street_negative_count,
            "street_neutral_or_weak_count": street_neutral_or_weak_count,
            "street_positive_score": round(street_positive_score, 3),
            "street_negative_score": round(street_negative_score, 3),
            "street_strong_negative_count": street_strong_negative_count,
            "street_strong_negative_score": round(street_strong_negative_score, 3),
        },
        "addressed_conflict_eids": addressed_conflicts,
        "unused_conflict_eids": unused_conflicts,
    }

    return result


