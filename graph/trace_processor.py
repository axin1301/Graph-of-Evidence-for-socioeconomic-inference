def extract_trace_features(reasoning_trace):
    """
    Convert reasoning trace into structured, solver-usable features.
    """
    if reasoning_trace is None:
        return {
            "has_trace": False,
            "initial_status": None,
            "final_status": None,
            "refinement_used": False,
            "reflection_triggered": False,
            "num_unused_conflicts_initial": 0,
            "num_addressed_conflicts_final": 0,
            "support_balance": 0.0,
            "claim_stability_delta": 0.0,
            "confidence_delta": 0.0,
        }

    initial_ver = reasoning_trace.get("initial_verification", {}) or {}
    final_ver = reasoning_trace.get("final_verification", {}) or {}
    refined_claim = reasoning_trace.get("refined_claim")
    initial_claim = reasoning_trace.get("initial_claim")

    initial_status = initial_ver.get("status")
    final_status = final_ver.get("status")

    initial_summary = initial_ver.get("support_summary", {}) or {}
    final_summary = final_ver.get("support_summary", {}) or {}

    initial_positive = float(initial_summary.get("positive_support_score", 0.0))
    initial_negative = float(initial_summary.get("negative_conflict_score", 0.0))

    support_balance = initial_positive - initial_negative

    initial_estimate = None
    refined_estimate = None
    initial_conf = None
    refined_conf = None

    if initial_claim is not None:
        initial_estimate = float(initial_claim.get("estimate", 0.0))
        initial_conf = float(initial_claim.get("confidence", 0.0))

    if refined_claim is not None:
        refined_estimate = float(refined_claim.get("estimate", 0.0))
        refined_conf = float(refined_claim.get("confidence", 0.0))

    claim_stability_delta = 0.0
    confidence_delta = 0.0

    if initial_estimate is not None and refined_estimate is not None:
        claim_stability_delta = refined_estimate - initial_estimate

    if initial_conf is not None and refined_conf is not None:
        confidence_delta = refined_conf - initial_conf

    return {
        "has_trace": True,
        "initial_status": initial_status,
        "final_status": final_status,
        "refinement_used": bool(reasoning_trace.get("refinement_used", False)),
        "reflection_triggered": bool(reasoning_trace.get("reflection_triggered", False)),
        "num_unused_conflicts_initial": len(initial_ver.get("unused_conflict_eids", []) or []),
        "num_addressed_conflicts_final": len(final_ver.get("addressed_conflict_eids", []) or []),
        "num_issues_initial": len(initial_ver.get("issues", []) or []),
        "num_issues_final": len(final_ver.get("issues", []) or []),
        "support_balance": support_balance,
        "claim_stability_delta": claim_stability_delta,
        "confidence_delta": confidence_delta,
        "initial_positive_support_score": initial_positive,
        "initial_negative_conflict_score": initial_negative,
        "final_support_count": int(final_summary.get("support_count", 0)),
        "final_satellite_support_count": int(final_summary.get("satellite_support_count", 0)),
        "final_street_support_count": int(final_summary.get("street_support_count", 0)),
    }

