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


def build_reflection_prompt(claim, evidences, verification_result, task_spec, query):
    target_field = task_spec["target_field"]
    task_name = task_spec["task_name"]
    normalized_range = task_spec["normalized_range"]

    evidence_lines = []
    for e in evidences:
        implication = e.implication.get(target_field, 0)
        evidence_lines.append(
            f"{e.eid} | {e.modality} | {e.observation} | {target_field} implication={implication} | confidence={e.confidence}"
        )
    evidence_block = "\n".join(evidence_lines)

    prompt = f"""
You are an expert socioeconomic analyst performing a final reflection step for an evidence-constrained reasoning system.

The current claim still fails verification. Revise it once more so that it better aligns with the available evidence and the reported issues.

Requirements:
- Use only the provided evidence
- Do not invent new observations
- Explicitly account for unresolved contradictions if they exist
- Keep the estimate within {normalized_range}
- Return exactly one JSON object

JSON format:
{{
  "hypothesis": "...",
  "estimate": 0.0,
  "confidence": 0.0,
  "support_eids": ["E_xxx"],
  "contradict_eids": ["E_xxx"]
}}

Task:
{task_name}

Current claim:
{claim.to_dict()}

Verification result:
{verification_result}

Evidence:
{evidence_block}

User query:
{query}
"""
    return prompt


def parse_reflection_output(raw_output, task_name):
    try:
        clean_text = extract_json(raw_output)
        data = json.loads(clean_text)

        claim = Claim(
            cid=f"C_{task_name.lower()}_reflect_0",
            task=task_name,
            hypothesis=data["hypothesis"],
            estimate=float(data["estimate"]),
            confidence=float(data["confidence"]),
            support_eids=data["support_eids"],
            contradict_eids=data.get("contradict_eids", [])
        )
        return claim

    except Exception as e:
        print("Reflection parse error:", e)
        print("RAW:", raw_output)
        return None


def reflection_agent(claim, evidences, verification_result, task_spec, query, use_mock=False):
    target_field = task_spec["target_field"]
    task_name = task_spec["task_name"]

    if use_mock:
        contradict_eids = verification_result.get("unused_conflict_eids", [])
        reflected_claim = Claim(
            cid=f"C_{task_name.lower()}_reflect_0",
            task=task_name,
            hypothesis=f"The area shows mixed evidence for {task_name}, with moderate positive signals but also non-negligible contradictory street-level conditions, suggesting a cautious mid-range estimate.",
            estimate=4.9,
            confidence=0.6,
            support_eids=claim.support_eids,
            contradict_eids=contradict_eids
        )
        return reflected_claim

    prompt = build_reflection_prompt(claim, evidences, verification_result, task_spec, query)
    raw_output = LLM(prompt)

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    return parse_reflection_output(raw_output, task_name)



