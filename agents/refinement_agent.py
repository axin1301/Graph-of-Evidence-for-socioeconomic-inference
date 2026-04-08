from core.llm_api import LLM
import json
import re
from core.schemas import Claim

def extract_json(text):
    text = text.strip()

    code_block = re.search(r"```json(.*?)```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def build_refinement_prompt(claim, evidences, verification_result, task_spec, query):
    target_field = task_spec["target_field"]
    task_name = task_spec["task_name"]
    normalized_range = task_spec["normalized_range"]

    evidence_lines = []
    for e in evidences:
        implication = e.implication.get(target_field, 0)
        evidence_lines.append(
            f"{e.eid} | {e.modality} | {e.observation} | {task_name} implication={implication} | confidence={e.confidence}"
        )
    evidence_block = "\n".join(evidence_lines)

    prompt = f"""
You are an expert socioeconomic analyst refining a {task_name} claim under explicit evidence constraints.

You are given:
1. an initial {task_name} claim
2. all available evidence items
3. a verification result indicating that the current claim has issues

Your job:
- revise the claim so that it better reflects both supporting and conflicting evidence
- if contradictory evidence exists, reduce overconfidence
- if needed, adjust the estimate
- do not invent new evidence
- keep the estimate within {normalized_range}

Return exactly one JSON object:

{{
  "hypothesis": "...",
  "estimate": 0.0,
  "confidence": 0.0,
  "support_eids": ["E_xxx", "E_xxx"],
  "contradict_eids": ["E_xxx"]
}}

Task:
{task_name}

Initial claim:
{claim.to_dict()}

Verification result:
{verification_result}

Evidence:
{evidence_block}

User query:
{query}
"""
    return prompt

def parse_refined_claim_output(raw_output,task_spec):
    target_field = task_spec["target_field"]
    task_name = task_spec["task_name"]

    try:
        clean_text = extract_json(raw_output)
        data = json.loads(clean_text)

        contradict_eids = data.get("contradict_eids", [])

        claim = Claim(
            cid="C_" +target_field +"_0",
            task=target_field,
            hypothesis=data["hypothesis"],
            estimate=float(data["estimate"]),
            confidence=float(data["confidence"]),
            support_eids=data["support_eids"],
            contradict_eids=contradict_eids
        )

        return claim, contradict_eids

    except Exception as e:
        print("Refinement parse error:", e)
        print("RAW:", raw_output)
        return None, []

def refinement_agent(
    claim,
    evidences,
    verification_result,
    query,
    task_spec,
    use_mock=False,
    use_contradiction_handling=True
):
    target_field = task_spec["target_field"]
    task_name = task_spec["task_name"]

    if use_mock:
        contradict_eids = verification_result.get("unused_conflict_eids", [])

        if not use_contradiction_handling:
            contradict_eids = []

        refined_claim = Claim(
            cid="C_" +target_field +"_0",
            task=target_field,
            hypothesis="The area appears to be a moderately active urban zone with mixed residential and commercial activity.",
            estimate=5.3,
            confidence=0.7,
            support_eids=["E_sat_0", "E_st_0", "E_st_1"],
            contradict_eids=contradict_eids
        )

        return refined_claim, contradict_eids

    prompt = build_refinement_prompt(claim, evidences, verification_result, task_spec, query)
    raw_output = LLM(prompt)

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]
    claim, contradict_eids = parse_refined_claim_output(raw_output,task_spec)

    if not use_contradiction_handling:
        claim.contradict_eids = []

    return claim, claim.contradict_eids



