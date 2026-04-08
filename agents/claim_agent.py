import json
import math
import re
from core.schemas import Claim
from core.llm_api import LLM
from core.task_guidance import canonicalize_task_guidance_key

def extract_json(text):
    text = text.strip()

    code_block = re.search(r"```json(.*?)```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def build_gdp_claim_prompt(evidences, query, task_spec):
    target_field = task_spec["target_field"]
    task_name = task_spec["task_name"]
    unit = task_spec.get("unit")
    normalized_range = task_spec.get("normalized_range")
    target_key = canonicalize_task_guidance_key(target_field)

    evidence_text = []
    for e in evidences:
        implication = e.implication.get(target_field, 0)
        evidence_text.append(
            f"{e.eid} | {e.modality} | {e.observation} | {task_name} implication={implication} | confidence={e.confidence}"
        )

    evidence_block = "\n".join(evidence_text)

    scale_type = "unknown"
    estimate_instruction = 'set "estimate" to null if no numeric scale is provided.'

    if normalized_range is not None and len(normalized_range) == 2:
        lo, hi = normalized_range
        scale_type = "normalized"
        estimate_instruction = f'the estimate must be numeric and fall within [{lo}, {hi}].'

    task_specific_claim_guidance = """
- Base the claim on the strongest consistent visual evidence, not on generic urbanity alone.
"""

    if target_key == "population":
        task_specific_claim_guidance = """
- For Population, prioritize area-level residential intensity, density, and built-up coverage over wealth or visual quality.
- Dense ordinary housing can imply high population even if it does not imply high GDP or high house price.
- Low-rise, spacious, or fragmented development should pull the estimate down.
"""
    elif target_key == "carbon":
        task_specific_claim_guidance = """
- For Carbon, focus on transport intensity, road use, industrial signals, paved surfaces, and overall activity footprint.
- High greenery, sparse use, limited transport, and weak infrastructure should pull the estimate down.
- Do not infer high carbon merely from being urban; intensity matters more than urban identity.
"""
    elif target_key == "houseprice":
        task_specific_claim_guidance = """
- For HousePrice, prioritize neighborhood desirability, maintenance quality, amenity quality, orderliness, and residential appeal.
- Generic urban density, ordinary apartment blocks, or road activity do not by themselves imply high house price.
- If street-level evidence is mixed or ordinary, keep the estimate near average rather than optimistic.
"""
    elif target_key == "bachelorratio":
        task_specific_claim_guidance = """
- For BachelorRatio, treat the task as indirect contextual inference rather than a direct demographic readout.
- Prioritize cues of educational and professional neighborhood context, such as maintained built environment, service richness, walkability, mixed-use vitality, campus-like structure, and higher-quality amenities.
- Industrial, weak-amenity, deteriorated, or fragmented environments should pull the estimate down.
"""
    elif target_key == "violentcrime":
        task_specific_claim_guidance = """
- For ViolentCrime, focus on visible environmental risk cues such as disorder, vacancy, deterioration, harsh infrastructure context, security barriers, and weak pedestrian comfort.
- Well-maintained, orderly, comfortable, and cared-for environments should pull the estimate down.
- Avoid demographic stereotypes and rely only on visible environmental conditions.
"""
    elif target_key == "buildheight":
        task_specific_claim_guidance = """
- For BuildHeight, prioritize directly visible physical height cues and large-scale morphology over wealth, activity, or neighborhood prestige.
- Taller structures, strong verticality, compact high-intensity cores, and strong street enclosure should pull the estimate up.
- Warehouses, detached structures, low-rise blocks, and wide spacing should pull the estimate down.
"""

    prompt = f"""
You are an expert socioeconomic analyst.

Your task is to produce exactly one claim about {task_name} for an urban area based only on the provided structured evidence.

Use only the given evidence.
Do not introduce new observations.
Do not assume the area is above average unless the evidence clearly supports it.
If the evidence is weak, mixed, sparse, or contradictory, prefer conservative estimates near or below average rather than optimistic ones.
Treat weakly informative images as weak evidence, not as positive evidence.

{task_specific_claim_guidance}

First determine the RELATIVE LEVEL of the target.
Use exactly one coarse label from:
- very_low
- low
- slightly_low
- around_average
- slightly_high
- high
- very_high

Then assign:
- level_score:
  - a continuous number in [0, 1]
  - it should be consistent with the coarse level
  - examples:
    - very_low: 0.05-0.18
    - low: 0.18-0.35
    - slightly_low: 0.35-0.48
    - around_average: 0.45-0.55
    - slightly_high: 0.52-0.65
    - high: 0.65-0.82
    - very_high: 0.82-0.95

Then produce:
- hypothesis
- estimate
- confidence
- support_eids

Rules:
- Interpret the score relative to the dataset-style normalized scale rather than absolute real-world units.
- Use this rough scale anchor:
  - 0.0-2.0: very low
  - 2.0-4.0: low
  - 4.0-6.0: around average
  - 6.0-8.0: high
  - 8.0-9.9: very high
- The hypothesis must be consistent with the chosen level.
- scale_type should be "{scale_type}".
- {estimate_instruction}
- confidence must be between 0 and 1.
- Avoid defaulting to the center unless the evidence is truly mixed or weak.
- Do not default to high or slightly_high merely because some urban structures are present.
- If most evidence items are weakly informative, ambiguous, or near-zero in implication, the level should stay around_average, slightly_low, low, or very_low unless there is clear counter-evidence.
- If evidence includes explicit negative signals or missing infrastructure cues, reflect them in the chosen level.
- support_eids must only include evidence IDs from the provided evidence.
- The numeric estimate must be consistent with the selected level. For example, "around_average" cannot correspond to an extreme low or extreme high value.

Return exactly one JSON object:

{{
  "hypothesis": "...",
  "level": "around_average",
  "level_score": 0.5,
  "estimate": 0.0,
  "scale_type": "{scale_type}",
  "unit": {json.dumps(unit)},
  "confidence": 0.0,
  "support_eids": ["E_xxx", "E_xxx"]
}}

Evidence:
{evidence_block}

User query:
{query}
"""
    return prompt

LEVEL_TO_SCORE = {
    "very_low": 0.1,
    "low": 0.25,
    "slightly_low": 0.42,
    "around_average": 0.5,
    "slightly_high": 0.58,
    "high": 0.75,
    "very_high": 0.9,
}

def _robust_carbon_street_signal(evidences, target_field):
    subset = [e for e in evidences if getattr(e, "modality", None) == "street"]
    if not subset:
        return None

    weighted_items = []
    for e in subset:
        implication = float((e.implication or {}).get(target_field, 0.0))
        conf = float(getattr(e, "confidence", 0.5))
        informativeness = getattr(e, "informativeness", None)
        try:
            informativeness = float(informativeness) if informativeness is not None else 0.5
        except Exception:
            informativeness = 0.5
        informativeness = max(0.0, min(1.0, informativeness))

        scene_type = str(getattr(e, "scene_type", "") or "").lower()
        if scene_type == "weakly_informative":
            informativeness *= 0.45

        weight = max(0.05, conf * (0.35 + 0.65 * informativeness))
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

def _compute_evidence_score(evidences, target_field):
    modality_scores = []
    target_key = canonicalize_task_guidance_key(target_field)

    for modality in ("satellite", "street"):
        subset = [e for e in evidences if getattr(e, "modality", None) == modality]
        if not subset:
            continue

        weighted_sum = 0.0
        weight_total = 0.0
        for e in subset:
            implication = float((e.implication or {}).get(target_field, 0.0))
            conf = float(getattr(e, "confidence", 0.5))
            informativeness = getattr(e, "informativeness", None)
            try:
                informativeness = float(informativeness) if informativeness is not None else 0.5
            except Exception:
                informativeness = 0.5
            informativeness = max(0.0, min(1.0, informativeness))

            scene_type = str(getattr(e, "scene_type", "") or "").lower()
            if scene_type == "weakly_informative":
                informativeness *= 0.6

            weight = max(0.05, conf * (0.35 + 0.65 * informativeness))
            weighted_sum += implication * weight
            weight_total += weight

        if weight_total > 0:
            modality_score = weighted_sum / weight_total

            # Task-specific modality weighting.
            if target_key == "gdp":
                modality_weight = 0.72 if modality == "satellite" else 0.28
            elif target_key == "population":
                modality_weight = 0.75 if modality == "satellite" else 0.25
            elif target_key == "houseprice":
                modality_weight = 0.15 if modality == "satellite" else 0.85
            elif target_key == "carbon":
                modality_weight = 0.60 if modality == "satellite" else 0.40
            elif target_key == "bachelorratio":
                modality_weight = 0.42 if modality == "satellite" else 0.58
            elif target_key == "violentcrime":
                modality_weight = 0.30 if modality == "satellite" else 0.70
            elif target_key == "buildheight":
                modality_weight = 0.68 if modality == "satellite" else 0.32
            else:
                modality_weight = 0.50

            modality_scores.append((modality, modality_score, modality_weight))

    if not modality_scores:
        return 0.5

    if target_key == "gdp":
        sat_score = next((score for modality, score, _ in modality_scores if modality == "satellite"), None)
        st_score = next((score for modality, score, _ in modality_scores if modality == "street"), None)

        if sat_score is None and st_score is None:
            mean_signal = 0.0
        elif sat_score is None:
            mean_signal = 0.65 * st_score
        elif st_score is None:
            mean_signal = sat_score
        else:
            # For GDP, satellite provides the area-level baseline and street
            # evidence acts as a bounded local correction rather than a peer.
            residual = st_score - sat_score
            residual = max(-0.22, min(0.22, residual))
            mean_signal = sat_score + 0.45 * residual
    elif target_key == "population":
        sat_score = next((score for modality, score, _ in modality_scores if modality == "satellite"), None)
        st_score = next((score for modality, score, _ in modality_scores if modality == "street"), None)

        if sat_score is None and st_score is None:
            mean_signal = 0.0
        elif sat_score is None:
            mean_signal = 0.75 * st_score
        elif st_score is None:
            mean_signal = sat_score
        else:
            residual = st_score - sat_score
            residual = max(-0.22, min(0.22, residual))
            mean_signal = sat_score + 0.42 * residual

        # For low-population areas, the more important cue is often the lack of
        # dense built-up structure. When both modalities are weak or sparse,
        # push the claim downward rather than leaving it near average.
        low_density_penalty = 0.0
        if sat_score is not None and sat_score < 0.10:
            low_density_penalty += min(0.12, 0.65 * (0.10 - sat_score))
        if st_score is not None and st_score < 0.02:
            low_density_penalty += min(0.07, 0.35 * (0.02 - st_score))
        if sat_score is not None and st_score is not None and sat_score < 0.12 and st_score < 0.05:
            low_density_penalty += 0.06

        mean_signal -= low_density_penalty
    elif target_key == "houseprice":
        sat_score = next((score for modality, score, _ in modality_scores if modality == "satellite"), None)
        st_score = next((score for modality, score, _ in modality_scores if modality == "street"), None)

        if sat_score is None and st_score is None:
            mean_signal = 0.0
        elif st_score is None:
            mean_signal = 0.45 * sat_score
        elif sat_score is None:
            mean_signal = 0.90 * st_score
        else:
            residual = sat_score - st_score
            residual = max(-0.14, min(0.14, residual))
            mean_signal = st_score + 0.20 * residual
    elif target_key == "carbon":
        sat_score = next((score for modality, score, _ in modality_scores if modality == "satellite"), None)
        st_score = _robust_carbon_street_signal(evidences, target_field)
        if st_score is None:
            st_score = next((score for modality, score, _ in modality_scores if modality == "street"), None)

        if sat_score is None and st_score is None:
            mean_signal = 0.0
        elif sat_score is None:
            mean_signal = 0.60 * st_score
        elif st_score is None:
            mean_signal = sat_score
        else:
            residual = st_score - sat_score
            residual = max(-0.14, min(0.14, residual))
            mean_signal = sat_score + 0.24 * residual

        # For carbon, ordinary urban streets should not automatically drag the
        # estimate toward the middle. Weak traffic and green residential scenes
        # should keep the claim lower unless satellite evidence clearly shows
        # strong built or industrial intensity.
        low_intensity_penalty = 0.0
        if sat_score is not None and sat_score < 0.12:
            low_intensity_penalty += min(0.10, 0.50 * (0.12 - sat_score))
        if st_score is not None and st_score < 0.08:
            low_intensity_penalty += min(0.06, 0.24 * (0.08 - st_score))
        if sat_score is not None and st_score is not None and sat_score < 0.15 and st_score < 0.10:
            low_intensity_penalty += 0.05

        mean_signal -= low_intensity_penalty
    elif target_key == "bachelorratio":
        sat_score = next((score for modality, score, _ in modality_scores if modality == "satellite"), None)
        st_score = next((score for modality, score, _ in modality_scores if modality == "street"), None)

        if sat_score is None and st_score is None:
            mean_signal = 0.0
        elif sat_score is None:
            mean_signal = 0.85 * st_score
        elif st_score is None:
            mean_signal = 0.60 * sat_score
        else:
            residual = st_score - sat_score
            residual = max(-0.18, min(0.18, residual))
            mean_signal = 0.35 * sat_score + 0.65 * st_score + 0.22 * residual

        low_context_penalty = 0.0
        if sat_score is not None and sat_score < 0.05:
            low_context_penalty += min(0.08, 0.45 * (0.05 - sat_score))
        if st_score is not None and st_score < 0.08:
            low_context_penalty += min(0.12, 0.55 * (0.08 - st_score))
        if sat_score is not None and st_score is not None and sat_score < 0.08 and st_score < 0.10:
            low_context_penalty += 0.05

        high_context_bonus = 0.0
        if st_score is not None and st_score > 0.20:
            high_context_bonus += min(0.10, 0.34 * (st_score - 0.20))
        if sat_score is not None and sat_score > 0.16:
            high_context_bonus += min(0.05, 0.18 * (sat_score - 0.16))
        if sat_score is not None and st_score is not None and sat_score > 0.14 and st_score > 0.18:
            high_context_bonus += 0.03
        if st_score is not None and st_score > 0.28:
            high_context_bonus += min(0.08, 0.28 * (st_score - 0.28))
        if sat_score is not None and st_score is not None and sat_score > 0.18 and st_score > 0.24:
            high_context_bonus += 0.05

        mean_signal = mean_signal - low_context_penalty + high_context_bonus
    elif target_key == "violentcrime":
        sat_score = next((score for modality, score, _ in modality_scores if modality == "satellite"), None)
        st_score = next((score for modality, score, _ in modality_scores if modality == "street"), None)

        if sat_score is None and st_score is None:
            mean_signal = 0.0
        elif st_score is None:
            mean_signal = 0.55 * sat_score
        elif sat_score is None:
            mean_signal = 0.90 * st_score
        else:
            residual = sat_score - st_score
            residual = max(-0.18, min(0.18, residual))
            mean_signal = st_score + 0.22 * residual
    elif target_key == "buildheight":
        sat_score = next((score for modality, score, _ in modality_scores if modality == "satellite"), None)
        st_score = next((score for modality, score, _ in modality_scores if modality == "street"), None)

        if sat_score is None and st_score is None:
            mean_signal = 0.0
        elif sat_score is None:
            mean_signal = 0.82 * st_score
        elif st_score is None:
            mean_signal = sat_score
        else:
            residual = st_score - sat_score
            residual = max(-0.14, min(0.14, residual))
            mean_signal = sat_score + 0.18 * residual

        low_rise_penalty = 0.0
        if sat_score is not None and sat_score < 0.06:
            low_rise_penalty += min(0.08, 0.40 * (0.06 - sat_score))
        if st_score is not None and st_score < 0.04:
            low_rise_penalty += min(0.05, 0.24 * (0.04 - st_score))
        if sat_score is not None and st_score is not None and sat_score < 0.08 and st_score < 0.06:
            low_rise_penalty += 0.03

        high_rise_bonus = 0.0
        if sat_score is not None and sat_score > 0.18:
            high_rise_bonus += min(0.08, 0.24 * (sat_score - 0.18))
        if st_score is not None and st_score > 0.14:
            high_rise_bonus += min(0.04, 0.16 * (st_score - 0.14))
        if sat_score is not None and st_score is not None and sat_score > 0.16 and st_score > 0.12:
            high_rise_bonus += 0.03

        mean_signal = mean_signal - low_rise_penalty + high_rise_bonus
    else:
        weighted_sum = sum(score * weight for _, score, weight in modality_scores)
        total_weight = sum(weight for _, _, weight in modality_scores)
        mean_signal = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Convert balanced multimodal evidence into a task-relative score.
    # The tanh mapping is smooth, bounded, and allows stronger separation
    # when evidence is consistently positive or negative.
    if target_key == "carbon":
        return max(0.0, min(1.0, 0.5 + 0.45 * math.tanh(1.6 * mean_signal)))
    if target_key == "bachelorratio":
        if mean_signal < -0.10:
            score = 0.5 + 0.56 * math.tanh(1.95 * mean_signal)
        elif mean_signal > 0.14:
            score = 0.5 + 0.60 * math.tanh(1.95 * mean_signal)
        else:
            score = 0.5 + 0.48 * math.tanh(1.70 * mean_signal)
        return max(0.0, min(1.0, score))
    if target_key == "buildheight":
        if mean_signal < -0.10:
            score = 0.5 + 0.54 * math.tanh(1.90 * mean_signal)
        elif mean_signal > 0.20:
            score = 0.5 + 0.64 * math.tanh(2.10 * mean_signal)
        elif mean_signal > 0.14:
            score = 0.5 + 0.60 * math.tanh(2.00 * mean_signal)
        else:
            score = 0.5 + 0.46 * math.tanh(1.70 * mean_signal)
        return max(0.0, min(1.0, score))

    return max(0.0, min(1.0, 0.5 + 0.45 * math.tanh(1.6 * mean_signal)))

def _level_to_bucket_score(level, level_score):
    if level_score is not None:
        return max(0.0, min(1.0, float(level_score)))
    return LEVEL_TO_SCORE.get(level, 0.5)

def _calibrate_claim_with_evidence(claim, evidences, task_spec):
    if claim is None:
        return None

    target_field = task_spec["target_field"]
    normalized_range = task_spec.get("normalized_range")
    evidence_score = _compute_evidence_score(evidences, target_field)

    bucket_score = _level_to_bucket_score(claim.level, claim.level_score)

    if claim.level_score is None:
        claim.level_score = 0.60 * evidence_score + 0.40 * bucket_score
    else:
        claim.level_score = max(0.0, min(1.0, 0.20 * float(claim.level_score) + 0.30 * bucket_score + 0.50 * evidence_score))

    if normalized_range is not None and len(normalized_range) == 2:
        lo, hi = normalized_range
        if hi > lo:
            if claim.estimate is not None:
                estimate_score = max(0.0, min(1.0, (float(claim.estimate) - lo) / (hi - lo)))
                calibrated_score = 0.15 * estimate_score + 0.30 * bucket_score + 0.55 * evidence_score
            else:
                calibrated_score = 0.35 * bucket_score + 0.65 * evidence_score

            claim.estimate = lo + calibrated_score * (hi - lo)

    return claim

def parse_claim_output(raw_output, task_spec):
    try:
        clean_text = extract_json(raw_output)
        data = json.loads(clean_text)

        level = data.get("level", None)
        raw_level_score = data.get("level_score", None)
        level_score = None

        try:
            if raw_level_score is not None:
                level_score = max(0.0, min(1.0, float(raw_level_score)))
        except Exception:
            level_score = None

        if level_score is None:
            level_score = LEVEL_TO_SCORE.get(level, None)

        estimate = data.get("estimate", None)
        if estimate is not None:
            try:
                estimate = float(estimate)
            except Exception:
                estimate = None

        normalized_range = task_spec.get("normalized_range", None)

        # keep estimate and level_score loosely aligned without snapping everything
        # into a few fixed buckets.
        if level_score is not None and normalized_range is not None and estimate is not None:
            lo, hi = normalized_range
            if hi > lo:
                estimate_score = (estimate - lo) / (hi - lo)
                estimate_score = max(0.0, min(1.0, estimate_score))

                if abs(estimate_score - level_score) > 0.35:
                    blended_score = 0.65 * estimate_score + 0.35 * level_score
                    estimate = lo + blended_score * (hi - lo)

        # if estimate missing but level exists, derive estimate from level
        if estimate is None and level_score is not None and normalized_range is not None:
            lo, hi = normalized_range
            estimate = lo + level_score * (hi - lo)

        claim = Claim(
            cid="C_" + str(task_spec["target_field"]) + "_0",
            task=task_spec["target_field"],
            hypothesis=data["hypothesis"],
            estimate=estimate,
            confidence=float(data["confidence"]),
            support_eids=data["support_eids"],
            contradict_eids=data.get("contradict_eids", []),
            level=level,
            level_score=level_score,
            scale_type=data.get("scale_type", "unknown"),
            unit=data.get("unit", task_spec.get("unit"))
        )
        return claim

    except Exception as e:
        print("Claim parse error:", e)
        print("RAW:", raw_output)
        return None

def gdp_claim_agent(evidences, query, task_spec, use_mock=False):

    prompt = build_gdp_claim_prompt(evidences, query, task_spec)
    raw_output = LLM(prompt)

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    claim = parse_claim_output(raw_output, task_spec)
    return _calibrate_claim_with_evidence(claim, evidences, task_spec)




