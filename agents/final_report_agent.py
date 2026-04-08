from core.llm_api import LLM
import json
import re

def extract_json(text):
    text = text.strip()

    code_block = re.search(r"```json(.*?)```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text

def build_final_report_prompt(goe, query, task_spec):
    task_name = task_spec["task_name"]
    normalized_range = task_spec["normalized_range"]

    solver = goe.get("solver_result", {})
    solved_claims = solver.get("solved_claims", [])
    solved_claim = solved_claims[0] if solved_claims else None
    factor_scores = solver.get("factor_scores", [])

    prompt = f"""
You are an expert socioeconomic analyst.

You are given:
1. A Graph-of-Evidence (GoE)
2. A structured graph solver result
3. A user query

Your task is to produce a final structured report grounded primarily in the solver result.

Rules:
- Use the solver result as the primary basis for final_estimate and confidence.
- Use the GoE only to reference evidence IDs and summarize supporting / contradicting evidence.
- Do not introduce new observations.
- The final estimate must remain within {normalized_range}.
- Acknowledge both supporting and contradicting evidence.
- Mention major latent factors if they are present in the solver result.
- Do not substantially override the solver result.

Return exactly one JSON object:

{{
  "task": "{task_name}",
  "final_estimate": 0.0,
  "confidence": 0.0,
  "summary": "...",
  "supporting_evidence": ["E_xxx"],
  "contradicting_evidence": ["E_xxx"],
  "uncertainty_note": "..."
}}

Solved claim:
{json.dumps(solved_claim, indent=2)}

Factor scores:
{json.dumps(factor_scores[:5], indent=2)}

User query:
{query}
"""
    return prompt


def parse_final_report(raw_output):
    try:
        clean_text = extract_json(raw_output)
        data = json.loads(clean_text)
        return data
    except Exception as e:
        print("Final report parse error:", e)
        print("RAW:", raw_output)
        return None

def final_report_agent(goe, query, task_spec, use_mock=False):
    task_name = task_spec["task_name"]

    solver = goe.get("solver_result", {})
    solved_claims = solver.get("solved_claims", [])
    solved_claim = solved_claims[0] if solved_claims else None

    if use_mock:
        if solved_claim is None:
            return {
                "task": task_name,
                "final_estimate": 5.0,
                "confidence": 0.5,
                "summary": f"No solver result available; returning a fallback report for {task_name}.",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "uncertainty_note": "Solver result missing."
            }

        return {
            "task": task_name,
            "final_estimate": solved_claim["solved_estimate"],
            "confidence": solved_claim["solved_confidence"],
            "summary": f"Structured graph inference indicates a {task_name} estimate supported by direct evidence and latent urban-factor activations.",
            "supporting_evidence": solved_claim.get("top_supporting_evidence", []),
            "contradicting_evidence": solved_claim.get("top_conflicting_evidence", []),
            "uncertainty_note": f"Uncertainty reflects graph-based calibration, contradiction handling, and verification-aware reasoning."
        }

    prompt = build_final_report_prompt(goe, query, task_spec)
    raw_output = LLM(prompt)

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    return parse_final_report(raw_output)

def final_report_from_solver(goe, task_spec):
    solver = goe["solver_result"]
    claim = solver["solved_claims"][0]

    return {
        "task": task_spec["task_name"],
        "final_estimate": claim["solved_estimate"],
        "confidence": claim["solved_confidence"],
        "summary": f"Structured graph inference indicates a {task_spec['task_name']} estimate supported by {len(claim['top_supporting_evidence'])} key evidence items, while conflicting evidence remains explicitly accounted for.",
        "supporting_evidence": claim["top_supporting_evidence"],
        "contradicting_evidence": claim["top_conflicting_evidence"],
        "uncertainty_note": f"Uncertainty reflects verification-aware calibration, including contradiction handling and refinement stability."
    }




