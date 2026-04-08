import math
from core.task_guidance import canonicalize_task_guidance_key


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def compute_factor_gate(factor_outputs, raw_factor_score, direct_score):
    """
    Global confidence gate for factor-mediated reasoning.

    Weak, isolated, or contradictory factor activations should not dominate the
    solver. Strong and consistent factor patterns still pass through.
    """
    if not factor_outputs:
        return 0.35

    scores = [float(f.get("score", 0.0) or 0.0) for f in factor_outputs]
    top_factor_score = max(scores) if scores else 0.0
    mean_factor_score = sum(scores) / len(scores) if scores else 0.0
    num_active_factors = len([s for s in scores if s > 0.2])

    gate = 1.0

    if num_active_factors == 0:
        gate *= 0.45
    elif num_active_factors == 1:
        gate *= 0.72

    if top_factor_score < 0.18:
        gate *= 0.75
    elif top_factor_score < 0.24:
        gate *= 0.88

    if mean_factor_score < 0.12:
        gate *= 0.82

    if abs(raw_factor_score) < 0.04:
        gate *= 0.78

    if abs(direct_score) > 0.08 and raw_factor_score * direct_score < 0:
        gate *= 0.55

    return clamp(gate, 0.25, 1.0)


def compute_adaptive_routing_weights(
    prior_score,
    direct_score,
    factor_score,
    trace,
    factor_outputs,
    direct_contribs
):
    """
    Adaptive routing over three channels:
    - prior
    - direct evidence
    - factor-mediated reasoning

    Returns weights that sum to 1.
    """

    # base gates
    g_prior = 1.0
    g_direct = sigmoid(2.0 * direct_score - 1.0)
    g_factor = sigmoid(2.0 * factor_score - 1.0)

    num_active_factors = len([f for f in factor_outputs if f.get("score", 0.0) > 0.2])
    num_direct_support = len([x for x in direct_contribs if x.get("net_contribution", 0.0) > 0])

    # strengthen factor channel if factor layer is clearly active
    if num_active_factors >= 2:
        g_factor += 0.15

    # strengthen direct channel if there are multiple positive direct supports
    if num_direct_support >= 3:
        g_direct += 0.10

    # trace-aware adjustments
    if trace.get("refinement_used", False):
        g_prior += 0.10
        g_direct -= 0.05
        g_factor -= 0.05

    if trace.get("reflection_triggered", False):
        g_prior += 0.12
        g_direct -= 0.04
        g_factor -= 0.04

    if trace.get("initial_status") == "FAIL":
        g_prior += 0.10

    if trace.get("final_status") == "PASS":
        g_direct += 0.05
        g_factor += 0.05

    # keep positive
    g_prior = max(0.05, g_prior)
    g_direct = max(0.05, g_direct)
    g_factor = max(0.05, g_factor)

    # for clearly weak-signal cases, reduce prior and allow lower prediction
    if direct_score < 0.15 and factor_score < 0.15:
        g_prior -= 0.20
        g_direct += 0.05
        g_factor += 0.05

    g_prior = max(0.05, g_prior)
    g_direct = max(0.05, g_direct)
    g_factor = max(0.05, g_factor)

    total = g_prior + g_direct + g_factor

    w_prior = g_prior / total
    w_direct = g_direct / total
    w_factor = g_factor / total

    return {
        "g_prior": round(g_prior, 4),
        "g_direct": round(g_direct, 4),
        "g_factor": round(g_factor, 4),
        "w_prior": round(w_prior, 4),
        "w_direct": round(w_direct, 4),
        "w_factor": round(w_factor, 4),
        "num_active_factors": num_active_factors,
        "num_direct_support": num_direct_support
    }

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def _normalize_estimate(estimate, normalized_range):
    lo, hi = normalized_range
    if hi <= lo:
        return 0.5
    return clamp((estimate - lo) / (hi - lo))

def _denormalize_score(score, normalized_range):
    lo, hi = normalized_range
    return lo + score * (hi - lo)

def _weighted_modality_signal(evidences, target_field, modality):
    subset = [e for e in evidences if e.get("modality") == modality]
    if not subset:
        return None

    weighted_sum = 0.0
    weight_total = 0.0

    for e in subset:
        implication = float((e.get("implication") or {}).get(target_field, 0.0))
        confidence = float(e.get("confidence", 0.5))

        informativeness = e.get("informativeness", 0.5)
        try:
            informativeness = float(informativeness)
        except Exception:
            informativeness = 0.5
        informativeness = clamp(informativeness)

        scene_type = str(e.get("scene_type", "") or "").lower()
        if scene_type == "weakly_informative":
            informativeness *= 0.45

        weight = max(0.05, confidence * (0.30 + 0.70 * informativeness))
        weighted_sum += implication * weight
        weight_total += weight

    if weight_total <= 0:
        return None
    return weighted_sum / weight_total

def _robust_carbon_street_signal(evidences, target_field):
    subset = [e for e in evidences if e.get("modality") == "street"]
    if not subset:
        return None

    weighted_items = []
    for e in subset:
        implication = float((e.get("implication") or {}).get(target_field, 0.0))
        confidence = float(e.get("confidence", 0.5))

        informativeness = e.get("informativeness", 0.5)
        try:
            informativeness = float(informativeness)
        except Exception:
            informativeness = 0.5
        informativeness = clamp(informativeness)

        scene_type = str(e.get("scene_type", "") or "").lower()
        if scene_type == "weakly_informative":
            informativeness *= 0.45

        weight = max(0.05, confidence * (0.30 + 0.70 * informativeness))
        weighted_items.append((implication, weight))

    total_weight = sum(weight for _, weight in weighted_items)
    if total_weight <= 0:
        return None

    weighted_mean = sum(imp * weight for imp, weight in weighted_items) / total_weight

    ordered = sorted(weighted_items, key=lambda x: x[0])
    cumulative = 0.0
    weighted_median = ordered[-1][0]
    for implication, weight in ordered:
        cumulative += weight
        if cumulative >= 0.5 * total_weight:
            weighted_median = implication
            break

    strong_positive_mass = sum(weight for imp, weight in weighted_items if imp >= 0.25) / total_weight
    strong_negative_mass = sum(weight for imp, weight in weighted_items if imp <= -0.20) / total_weight

    signal = 0.70 * weighted_median + 0.30 * weighted_mean
    if signal > 0 and strong_positive_mass < 0.30:
        signal *= 0.75
    if signal < 0 and strong_negative_mass < 0.25:
        signal *= 0.85
    return signal

def _get_evidence_anchor(goe_aug):
    aggregation = goe_aug.get("aggregation", {}) or {}
    target_field = goe_aug.get("meta", {}).get("target_field")
    evidences = goe_aug.get("evidences", []) or []
    target_key = canonicalize_task_guidance_key(target_field)

    if target_key == "gdp" and evidences:
        sat_signal = _weighted_modality_signal(evidences, target_field, "satellite")
        st_signal = _weighted_modality_signal(evidences, target_field, "street")

        if sat_signal is None and st_signal is None:
            return 0.5
        if sat_signal is None:
            combined_signal = 0.60 * st_signal
        elif st_signal is None:
            combined_signal = sat_signal
        else:
            # GDP should be anchored by the area-level satellite pattern.
            # Street evidence only nudges the estimate within a narrow band.
            street_residual = st_signal - sat_signal
            street_residual = max(-0.20, min(0.20, street_residual))
            combined_signal = sat_signal + 0.40 * street_residual

        anchor = 0.5 + 0.34 * math.tanh(1.45 * combined_signal)
        return clamp(anchor)
    if target_key == "population" and evidences:
        sat_signal = _weighted_modality_signal(evidences, target_field, "satellite")
        st_signal = _weighted_modality_signal(evidences, target_field, "street")

        if sat_signal is None and st_signal is None:
            return 0.5
        if sat_signal is None:
            combined_signal = 0.75 * st_signal
        elif st_signal is None:
            combined_signal = sat_signal
        else:
            street_residual = st_signal - sat_signal
            street_residual = max(-0.20, min(0.20, street_residual))
            combined_signal = sat_signal + 0.38 * street_residual

        low_density_penalty = 0.0
        if sat_signal is not None and sat_signal < 0.10:
            low_density_penalty += min(0.12, 0.65 * (0.10 - sat_signal))
        if st_signal is not None and st_signal < 0.02:
            low_density_penalty += min(0.07, 0.35 * (0.02 - st_signal))
        if sat_signal is not None and st_signal is not None and sat_signal < 0.12 and st_signal < 0.05:
            low_density_penalty += 0.06

        combined_signal -= low_density_penalty

        anchor = 0.5 + 0.38 * math.tanh(1.55 * combined_signal)
        return clamp(anchor)
    if target_key == "houseprice" and evidences:
        sat_signal = _weighted_modality_signal(evidences, target_field, "satellite")
        st_signal = _weighted_modality_signal(evidences, target_field, "street")

        if sat_signal is None and st_signal is None:
            return 0.5
        if st_signal is None:
            combined_signal = 0.45 * sat_signal
        elif sat_signal is None:
            combined_signal = 0.90 * st_signal
        else:
            sat_residual = sat_signal - st_signal
            sat_residual = max(-0.12, min(0.12, sat_residual))
            combined_signal = st_signal + 0.18 * sat_residual

        anchor = 0.5 + 0.30 * math.tanh(1.35 * combined_signal)
        return clamp(anchor)
    if target_key == "carbon" and evidences:
        sat_signal = _weighted_modality_signal(evidences, target_field, "satellite")
        st_signal = _robust_carbon_street_signal(evidences, target_field)
        if st_signal is None:
            st_signal = _weighted_modality_signal(evidences, target_field, "street")

        if sat_signal is None and st_signal is None:
            return 0.5
        if sat_signal is None:
            combined_signal = 0.65 * st_signal
        elif st_signal is None:
            combined_signal = sat_signal
        else:
            street_residual = st_signal - sat_signal
            street_residual = max(-0.14, min(0.14, street_residual))
            combined_signal = sat_signal + 0.24 * street_residual

        low_intensity_penalty = 0.0
        if sat_signal is not None and sat_signal < 0.12:
            low_intensity_penalty += min(0.10, 0.50 * (0.12 - sat_signal))
        if st_signal is not None and st_signal < 0.08:
            low_intensity_penalty += min(0.06, 0.24 * (0.08 - st_signal))
        if sat_signal is not None and st_signal is not None and sat_signal < 0.15 and st_signal < 0.10:
            low_intensity_penalty += 0.05

        combined_signal -= low_intensity_penalty

        anchor = 0.5 + 0.36 * math.tanh(1.45 * combined_signal)
        return clamp(anchor)
    if target_key == "bachelorratio" and evidences:
        sat_signal = _weighted_modality_signal(evidences, target_field, "satellite")
        st_signal = _weighted_modality_signal(evidences, target_field, "street")

        if sat_signal is None and st_signal is None:
            return 0.5
        if sat_signal is None:
            combined_signal = 0.85 * st_signal
        elif st_signal is None:
            combined_signal = 0.60 * sat_signal
        else:
            residual = st_signal - sat_signal
            residual = max(-0.18, min(0.18, residual))
            combined_signal = 0.35 * sat_signal + 0.65 * st_signal + 0.18 * residual

        low_context_penalty = 0.0
        if sat_signal is not None and sat_signal < 0.05:
            low_context_penalty += min(0.08, 0.45 * (0.05 - sat_signal))
        if st_signal is not None and st_signal < 0.08:
            low_context_penalty += min(0.12, 0.55 * (0.08 - st_signal))
        if sat_signal is not None and st_signal is not None and sat_signal < 0.08 and st_signal < 0.10:
            low_context_penalty += 0.05

        high_context_bonus = 0.0
        if st_signal is not None and st_signal > 0.20:
            high_context_bonus += min(0.10, 0.34 * (st_signal - 0.20))
        if sat_signal is not None and sat_signal > 0.16:
            high_context_bonus += min(0.05, 0.18 * (sat_signal - 0.16))
        if sat_signal is not None and st_signal is not None and sat_signal > 0.14 and st_signal > 0.18:
            high_context_bonus += 0.03
        if st_signal is not None and st_signal > 0.28:
            high_context_bonus += min(0.08, 0.28 * (st_signal - 0.28))
        if sat_signal is not None and st_signal is not None and sat_signal > 0.18 and st_signal > 0.24:
            high_context_bonus += 0.05

        combined_signal = combined_signal - low_context_penalty + high_context_bonus

        if combined_signal < -0.10:
            anchor = 0.5 + 0.54 * math.tanh(1.90 * combined_signal)
        elif combined_signal > 0.14:
            anchor = 0.5 + 0.58 * math.tanh(1.90 * combined_signal)
        else:
            anchor = 0.5 + 0.40 * math.tanh(1.55 * combined_signal)
        return clamp(anchor)
    if target_key == "violentcrime" and evidences:
        sat_signal = _weighted_modality_signal(evidences, target_field, "satellite")
        st_signal = _weighted_modality_signal(evidences, target_field, "street")

        if sat_signal is None and st_signal is None:
            return 0.5
        if st_signal is None:
            combined_signal = 0.55 * sat_signal
        elif sat_signal is None:
            combined_signal = 0.90 * st_signal
        else:
            residual = sat_signal - st_signal
            residual = max(-0.16, min(0.16, residual))
            combined_signal = st_signal + 0.20 * residual

        anchor = 0.5 + 0.34 * math.tanh(1.40 * combined_signal)
        return clamp(anchor)
    if target_key == "buildheight" and evidences:
        sat_signal = _weighted_modality_signal(evidences, target_field, "satellite")
        st_signal = _weighted_modality_signal(evidences, target_field, "street")

        if sat_signal is None and st_signal is None:
            return 0.5
        if sat_signal is None:
            combined_signal = 0.82 * st_signal
        elif st_signal is None:
            combined_signal = sat_signal
        else:
            residual = st_signal - sat_signal
            residual = max(-0.12, min(0.12, residual))
            combined_signal = sat_signal + 0.18 * residual

        low_rise_penalty = 0.0
        if sat_signal is not None and sat_signal < 0.06:
            low_rise_penalty += min(0.08, 0.40 * (0.06 - sat_signal))
        if st_signal is not None and st_signal < 0.04:
            low_rise_penalty += min(0.05, 0.24 * (0.04 - st_signal))
        if sat_signal is not None and st_signal is not None and sat_signal < 0.08 and st_signal < 0.06:
            low_rise_penalty += 0.03

        high_rise_bonus = 0.0
        if sat_signal is not None and sat_signal > 0.18:
            high_rise_bonus += min(0.10, 0.30 * (sat_signal - 0.18))
        if st_signal is not None and st_signal > 0.14:
            high_rise_bonus += min(0.04, 0.16 * (st_signal - 0.14))
        if sat_signal is not None and st_signal is not None and sat_signal > 0.16 and st_signal > 0.12:
            high_rise_bonus += 0.04

        combined_signal = combined_signal - low_rise_penalty + high_rise_bonus
        if combined_signal > 0.20:
            anchor = 0.5 + 0.50 * math.tanh(1.90 * combined_signal)
        else:
            anchor = 0.5 + 0.44 * math.tanh(1.70 * combined_signal)
        return clamp(anchor)

    net_score = None
    if isinstance(aggregation.get("balanced"), dict):
        net_score = aggregation["balanced"].get("net_score")
    if net_score is None and isinstance(aggregation.get("overall"), dict):
        net_score = aggregation["overall"].get("net_score")
    if net_score is None:
        return 0.5

    try:
        net_score = float(net_score)
    except Exception:
        return 0.5

    # Map evidence net score to [0,1] with a steeper response on the low side,
    # so clearly weak or negative evidence can pull predictions down.
    anchor = 0.5 + 0.38 * math.tanh(1.35 * net_score)
    return clamp(anchor)

def _get_bucket_anchor(claim, normalized_range):
    level_score = claim.get("level_score", None)
    if level_score is not None:
        try:
            return clamp(float(level_score))
        except Exception:
            pass

    estimate = claim.get("estimate", None)
    if estimate is not None:
        try:
            return _normalize_estimate(float(estimate), normalized_range)
        except Exception:
            pass

    return 0.5

def solve_goe_claims_v3(goe_aug, edge_potential_result, task_spec,
    factor_prior_scale=0.7,
    redundancy_penalty_scale=0.05,
    verification_bonus_scale=0.5,
    estimate_temperature=1.1):

    target_field = task_spec["target_field"]
    norm_range = task_spec["normalized_range"]

    claims = goe_aug.get("claims", [])
    evidences = {e["id"]: e for e in goe_aug.get("evidences", [])}
    factors = {f["id"]: f for f in goe_aug.get("factors", [])}
    relations = {r["id"]: r for r in goe_aug.get("relations", [])}
    trace = goe_aug.get("goe_trace", {}) or {}
    trace = dict(trace)
    trace["target_field"] = target_field
    evidence_anchor = _get_evidence_anchor(goe_aug)
    target_key = canonicalize_task_guidance_key(target_field)

    potentials = edge_potential_result.get("potentials", [])

    # ---------------------------
    # 1) evidence calibration
    # ---------------------------
    redundancy_penalty = {eid: 0.0 for eid in evidences.keys()}
    source_reliability_acc = {eid: [] for eid in evidences.keys()}

    for p in potentials:
        edge_id = p["edge_id"]
        rel = relations.get(edge_id)
        if rel is None:
            continue

        if rel["relation_type"] == "candidate_redundancy":
            red = float(p.get("redundancy_strength", 0.0))
            from_id = rel["from_id"]
            to_id = rel["to_id"]
            redundancy_penalty[from_id] += redundancy_penalty_scale * red
            redundancy_penalty[to_id] += redundancy_penalty_scale * red
        if rel["from_id"] in evidences and rel["relation_type"] in (
            "support", "contradict", "candidate_evidence_claim",
            "candidate_evidence_factor", "evidence_factor"
        ):
            source_reliability_acc[rel["from_id"]].append(float(p.get("source_reliability", 0.5)))

    calibrated_evidences = {}
    for eid, e in evidences.items():
        base_conf = float(e.get("confidence", 0.5))
        src_rel = sum(source_reliability_acc[eid]) / len(source_reliability_acc[eid]) if source_reliability_acc[eid] else 0.55

        calibrated = (
            0.75 * base_conf +
            0.15 * src_rel -
            redundancy_penalty[eid]
        )
        calibrated = clamp(calibrated)

        calibrated_evidences[eid] = {
            "evidence_id": eid,
            "original_confidence": round(base_conf, 4),
            "source_reliability": round(src_rel, 4),
            "redundancy_penalty": round(redundancy_penalty[eid], 4),
            "calibrated_confidence": round(calibrated, 4),
            "modality": e.get("modality"),
            "observation": e.get("observation"),
        }

    # ---------------------------
    # 2) factor activation
    # ---------------------------
    factor_scores = {fid: 0.0 for fid in factors.keys()}
    factor_evidence_links = {fid: [] for fid in factors.keys()}

    for p in potentials:
        edge_id = p["edge_id"]
        rel = relations.get(edge_id)
        if rel is None:
            continue
        if rel["relation_type"] in ("candidate_evidence_factor", "evidence_factor"):
            eid = rel["from_id"]
            fid = rel["to_id"]

            if eid in evidences and fid in factors:
                e_conf = calibrated_evidences[eid]["calibrated_confidence"]
                s = float(p.get("support_strength", 0.0))
                contrib = s * e_conf

                factor_scores[fid] += contrib
                factor_evidence_links[fid].append({
                    "evidence_id": eid,
                    "contribution": round(contrib, 4),
                    "support_strength": round(s, 4),
                    "evidence_confidence": round(e_conf, 4)
                })

    factor_outputs = []
    for fid, f in factors.items():
        raw = factor_scores[fid]
        score = raw / (1.0 + raw)

        ranked = sorted(
            factor_evidence_links[fid],
            key=lambda x: x["contribution"],
            reverse=True
        )

        factor_outputs.append({
            "factor_id": fid,
            "factor_name": f["factor_name"],
            "score": round(score, 4),
            "top_evidence": [x["evidence_id"] for x in ranked[:5]],
            "evidence_links": ranked
        })

    factor_output_index = {x["factor_id"]: x for x in factor_outputs}

    # ---------------------------
    # 3) direct evidence -> claim contribution
    # ---------------------------
    evidence_contrib_by_claim = {c["id"]: [] for c in claims}
    factor_contrib_by_claim = {c["id"]: [] for c in claims}

    for p in potentials:
        edge_id = p["edge_id"]
        rel = relations.get(edge_id)
        if rel is None:
            continue

        from_id = rel["from_id"]
        to_id = rel["to_id"]
        rtype = rel["relation_type"]

        if from_id in evidences and to_id in evidence_contrib_by_claim and rtype in ("support", "contradict", "candidate_evidence_claim"):
            e_conf = calibrated_evidences[from_id]["calibrated_confidence"]
            support_strength = float(p.get("support_strength", 0.0))
            conflict_strength = float(p.get("conflict_strength", 0.0))
            negative_boost = 1.25
            modality = evidences[from_id].get("modality")

            support_scale = 1.0
            conflict_scale = negative_boost

            if target_key == "gdp" and modality == "street":
                scene_type = str(evidences[from_id].get("scene_type", "") or "").lower()
                informativeness = evidences[from_id].get("informativeness", 0.5)
                try:
                    informativeness = float(informativeness)
                except Exception:
                    informativeness = 0.5
                informativeness = clamp(informativeness)

                support_scale = 0.55
                conflict_scale = 1.45

                if scene_type == "weakly_informative":
                    support_scale *= 0.45
                    conflict_scale *= 0.90
                elif informativeness < 0.45:
                    support_scale *= 0.70
            elif target_key == "population":
                if modality == "satellite":
                    support_scale = 1.05
                    conflict_scale = 1.15
                else:
                    support_scale = 0.75
                    conflict_scale = 1.20
            elif target_key == "houseprice":
                if modality == "satellite":
                    support_scale = 0.30
                    conflict_scale = 0.85
                else:
                    support_scale = 1.10
                    conflict_scale = 1.20
            elif target_key == "carbon":
                if modality == "satellite":
                    support_scale = 1.00
                    conflict_scale = 1.15
                else:
                    support_scale = 0.72
                    conflict_scale = 1.15
            elif target_key == "bachelorratio":
                if modality == "satellite":
                    support_scale = 0.85
                    conflict_scale = 1.05
                else:
                    support_scale = 1.00
                    conflict_scale = 1.10
            elif target_key == "violentcrime":
                if modality == "satellite":
                    support_scale = 0.75
                    conflict_scale = 1.00
                else:
                    support_scale = 1.10
                    conflict_scale = 1.15
            elif target_key == "buildheight":
                if modality == "satellite":
                    support_scale = 1.05
                    conflict_scale = 1.10
                else:
                    support_scale = 0.95
                    conflict_scale = 1.05

            net = (support_scale * support_strength - conflict_scale * conflict_strength) * e_conf

            evidence_contrib_by_claim[to_id].append({
                "edge_id": edge_id,
                "evidence_id": from_id,
                "support_strength": round(support_strength, 4),
                "conflict_strength": round(conflict_strength, 4),
                "evidence_confidence": round(e_conf, 4),
                "net_contribution": round(net, 4),
            })

        elif from_id in factors and to_id in factor_contrib_by_claim and rtype == "factor_claim":
            factor_score = factor_output_index[from_id]["score"]
            support_strength = float(p.get("support_strength", 0.0))
            conflict_strength = float(p.get("conflict_strength", 0.0))
            # Weak factor activations should stay near neutral rather than
            # systematically pushing claims upward. Only sufficiently activated
            # factors contribute strongly to the final claim.
            if target_key == "houseprice":
                effective_factor_score = max(0.0, (factor_score - 0.08) / 0.92)
            elif target_key == "buildheight":
                effective_factor_score = max(0.0, (factor_score - 0.10) / 0.90)
            else:
                effective_factor_score = max(0.0, (factor_score - 0.18) / 0.82)
            net = (support_strength - conflict_strength) * effective_factor_score

            factor_contrib_by_claim[to_id].append({
                "edge_id": edge_id,
                "factor_id": from_id,
                "factor_name": factor_output_index[from_id]["factor_name"],
                "factor_score": round(factor_score, 4),
                "effective_factor_score": round(effective_factor_score, 4),
                "support_strength": round(support_strength, 4),
                "conflict_strength": round(conflict_strength, 4),
                "net_contribution": round(net, 4)
            })

    # ---------------------------
    # 4) claim solving + trace-aware calibration
    # ---------------------------
    solved_claims = []

    for c in claims:
        cid = c["id"]
        original_estimate = float(c.get("estimate", 0.0))
        original_confidence = float(c.get("confidence", 0.5))
        bucket_anchor = _get_bucket_anchor(c, norm_range)
        
        level_score = c.get("level_score", None)
        estimate_value = c.get("estimate", None)

        if level_score is not None and estimate_value is not None:
            estimate_score = _normalize_estimate(float(estimate_value), norm_range)
            prior_score = 0.5 * float(level_score) + 0.5 * estimate_score
        elif level_score is not None:
            prior_score = float(level_score)
        elif estimate_value is not None:
            prior_score = _normalize_estimate(float(estimate_value), norm_range)
        else:
            prior_score = 0.5

        direct_contribs = evidence_contrib_by_claim[cid]
        factor_contribs = factor_contrib_by_claim[cid]

        direct_raw = sum(x["net_contribution"] for x in direct_contribs)
        factor_raw = sum(x["net_contribution"] for x in factor_contribs)

        direct_score = direct_raw / (1.0 + abs(direct_raw))
        raw_factor_score = factor_raw / (1.0 + abs(factor_raw))
        factor_gate = compute_factor_gate(factor_outputs, raw_factor_score, direct_score)
        factor_score = raw_factor_score * factor_gate

        weak_evidence = abs(direct_score) < 0.10 and abs(factor_score) < 0.12
        high_prior_disagrees = prior_score - evidence_anchor > 0.12

        if weak_evidence:
            prior_score = 0.20 * prior_score + 0.35 * bucket_anchor + 0.45 * evidence_anchor
        elif high_prior_disagrees:
            prior_score = 0.40 * prior_score + 0.30 * bucket_anchor + 0.30 * evidence_anchor
        else:
            prior_score = 0.55 * prior_score + 0.25 * bucket_anchor + 0.20 * evidence_anchor

        routing = compute_adaptive_routing_weights(
            prior_score=prior_score,
            direct_score=direct_score,
            factor_score=factor_score,
            trace=trace,
            factor_outputs=factor_outputs,
            direct_contribs=direct_contribs
        )

        # Use graph channels as residual corrections around the prior estimate
        # instead of mapping everything back toward 0.5.
        graph_score = clamp(
            prior_score +
            0.95 * routing["w_direct"] * direct_score +
            0.55 * routing["w_factor"] * factor_score
        )

        verification_bonus = 0.0
        uncertainty_penalty = 0.0

        if trace.get("initial_status") == "FAIL":
            uncertainty_penalty += 0.08 * verification_bonus_scale

        if trace.get("final_status") == "PASS":
            verification_bonus += 0.03 * verification_bonus_scale

        if trace.get("num_unused_conflicts_initial", 0) > 0:
            uncertainty_penalty += min(0.12, 0.02 * trace["num_unused_conflicts_initial"]) * verification_bonus_scale

        if trace.get("num_addressed_conflicts_final", 0) > 0:
            verification_bonus += min(0.08, 0.015 * trace["num_addressed_conflicts_final"]) * verification_bonus_scale

        if trace.get("refinement_used", False):
            uncertainty_penalty += 0.03 * verification_bonus_scale

        if trace.get("reflection_triggered", False):
            uncertainty_penalty += 0.04 * verification_bonus_scale

        stability_shift = abs(float(trace.get("claim_stability_delta", 0.0)))
        uncertainty_penalty += min(0.08, 0.02 * stability_shift) * verification_bonus_scale
        mean_factor_score = 0.0
        if len(factor_outputs) > 0:
            mean_factor_score = sum(f["score"] for f in factor_outputs) / len(factor_outputs)

        num_active_factors = len([f for f in factor_outputs if f["score"] > 0.2])

        low_activation_penalty = 0.0
        if num_active_factors <= 1 and mean_factor_score < 0.15:
            low_activation_penalty = 0.02

        disagreement_penalty = 0.0
        weak_signal = direct_score < 0.08 and factor_score < 0.10
        if weak_signal and prior_score > 0.58:
            disagreement_penalty += min(0.20, 0.35 * (prior_score - 0.58))
        if direct_score < -0.05 and prior_score > 0.55:
            disagreement_penalty += min(0.15, 0.20 * abs(direct_score) + 0.12 * (prior_score - 0.55))
        if evidence_anchor < 0.35 and prior_score > evidence_anchor:
            disagreement_penalty += min(0.22, 0.45 * (prior_score - evidence_anchor))

        evidence_supported_high = (
            evidence_anchor >= 0.62
            and direct_score >= 0.12
            and (factor_score >= 0.05 or routing.get("num_direct_support", 0) >= 3)
        )
        if evidence_supported_high:
            uncertainty_penalty *= 0.78
            disagreement_penalty *= 0.70

        mean_factor_score = 0.0
        if len(factor_outputs) > 0:
            mean_factor_score = sum(f["score"] for f in factor_outputs) / len(factor_outputs)

        num_active_factors = len([f for f in factor_outputs if f["score"] > 0.2])

        low_value_bonus = 0.0
        if evidence_anchor < 0.30 and direct_score <= 0.05:
            low_value_bonus += min(0.18, 0.40 * (0.30 - evidence_anchor))
        elif evidence_anchor < 0.40 and direct_score <= 0.10:
            low_value_bonus += min(0.10, 0.22 * (0.40 - evidence_anchor))

        if target_key == "population":
            if evidence_anchor < 0.42 and direct_score <= 0.12:
                low_value_bonus += min(0.08, 0.20 * (0.42 - evidence_anchor))
            if evidence_anchor < 0.35 and factor_score <= 0.08:
                low_value_bonus += min(0.05, 0.18 * (0.35 - evidence_anchor))

        final_score = clamp(
            graph_score
            + verification_bonus
            - uncertainty_penalty
            - low_activation_penalty
            - disagreement_penalty
            - low_value_bonus
        )

        if weak_evidence:
            final_score = clamp(0.35 * final_score + 0.25 * bucket_anchor + 0.40 * evidence_anchor)
        else:
            final_score = clamp(0.70 * final_score + 0.20 * bucket_anchor + 0.10 * evidence_anchor)

        # ---------------------------
        # (1) temperature scaling
        # ---------------------------
        centered = final_score - 0.5
        final_score = clamp(0.5 + estimate_temperature * centered)

        # ---------------------------
        # (2) piecewise shrink for low-value region ⭐
        # ---------------------------
        if final_score < 0.18:
            final_score = clamp(final_score * 0.92)
        elif final_score < 0.35:
            final_score = clamp(final_score * 0.96)
        elif final_score > 0.82:
            final_score = clamp(0.08 + 0.92 * final_score)
        elif final_score > 0.65:
            final_score = clamp(0.04 + 0.96 * final_score)

        support_margin = max(0.0, direct_score + factor_score)
        final_confidence = clamp(
            0.45 * original_confidence +
            0.20 * (1.0 if trace.get("final_status") == "PASS" else 0.5) +
            0.15 * min(1.0, support_margin) +
            0.10 * (1.0 - min(1.0, abs(trace.get("confidence_delta", 0.0)))) -
            0.10 * (1.0 if trace.get("reflection_triggered", False) else 0.0)
        )

        top_support = sorted(
            [x for x in direct_contribs if x["net_contribution"] > 0],
            key=lambda x: x["net_contribution"],
            reverse=True
        )
        top_conflict = sorted(
            [x for x in direct_contribs if x["net_contribution"] < 0],
            key=lambda x: x["net_contribution"]
        )
        top_factors = sorted(
            factor_contribs,
            key=lambda x: x["net_contribution"],
            reverse=True
        )

        solved_claims.append({
            "claim_id": cid,
            "task": c.get("task"),
            "hypothesis": c.get("hypothesis"),
            "original_estimate": original_estimate,
            "original_confidence": original_confidence,
            "prior_score": round(prior_score, 4),
            "bucket_anchor": round(bucket_anchor, 4),
            "evidence_anchor": round(evidence_anchor, 4),
            "direct_evidence_score": round(direct_score, 4),
            "factor_graph_score": round(factor_score, 4),
            "raw_factor_graph_score": round(raw_factor_score, 4),
            "factor_gate": round(factor_gate, 4),
            "graph_score": round(graph_score, 4),
            "routing_weights": routing,
            "verification_bonus": round(verification_bonus, 4),
            "uncertainty_penalty": round(uncertainty_penalty, 4),
            "disagreement_penalty": round(disagreement_penalty, 4),
            "low_value_bonus": round(low_value_bonus, 4),
            "solved_claim_score": round(final_score, 4),
            "solved_estimate": round(_denormalize_score(final_score, norm_range), 4),
            "solved_confidence": round(final_confidence, 4),
            "top_supporting_evidence": [x["evidence_id"] for x in top_support[:5]],
            "top_conflicting_evidence": [x["evidence_id"] for x in top_conflict[:5]],
            "top_factors": [x["factor_name"] for x in top_factors[:5]],
            "direct_contributions": direct_contribs,
            "factor_contributions": factor_contribs
        })

    return {
        "method": "hierarchical_verification_aware_solver_v3",
        "target_field": target_field,
        "normalized_range": norm_range,
        "trace_features_used": trace,
        "calibrated_evidences": calibrated_evidences,
        "factor_scores": factor_outputs,
        "solved_claims": solved_claims,
        "config": {
            "factor_prior_scale": factor_prior_scale,
            "redundancy_penalty_scale": redundancy_penalty_scale,
            "verification_bonus_scale": verification_bonus_scale,
            "estimate_temperature": estimate_temperature
        },
    }




