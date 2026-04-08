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


def build_task_parser_prompt(query):
    prompt = f"""
You are an expert parser for urban socioeconomic inference tasks.

Given a user query, extract the task specification into a fixed JSON object.

Return exactly one JSON object with this format:

{{
  "task_name": "...",
  "target_field": "...",
  "unit": "...",
  "normalized_range": [0.0, 9.9]
}}

Rules:
- task_name should be a concise task label, e.g. "GDP", "Carbon", "HousePrice".
- target_field should match task_name.
- unit should be copied from the query if available, otherwise use null.
- normalized_range should be extracted from the query if explicitly stated.
- If the query asks for normalized output in 0-9.9, return [0.0, 9.9].
- Do not include any explanation.

User query:
{query}
"""
    return prompt


def _canonicalize_task_spec(task_spec):
    if task_spec is None:
        return {
            "task_name": "GDP",
            "target_field": "GDP",
            "unit": None,
            "normalized_range": [0.0, 9.9]
        }

    task_name_raw = str(task_spec.get("task_name", "") or "")
    target_field_raw = str(task_spec.get("target_field", "") or "")
    unit = task_spec.get("unit")
    normalized_range = task_spec.get("normalized_range", [0.0, 9.9])

    text = f"{task_name_raw} {target_field_raw}".lower()

    canonical_task = task_name_raw.strip() or "GDP"
    canonical_target = target_field_raw.strip() or canonical_task

    if "gdp" in text:
        canonical_task = "GDP"
        canonical_target = "GDP"
    elif "population" in text or re.search(r"\bpop\b", text):
        canonical_task = "Population"
        canonical_target = "Population"
    elif "carbon" in text or "co2" in text or "emission" in text:
        canonical_task = "Carbon"
        canonical_target = "Carbon"
    elif "house" in text and "price" in text:
        canonical_task = "HousePrice"
        canonical_target = "HousePrice"
    elif "bachelor" in text:
        canonical_task = "BachelorRatio"
        canonical_target = "BachelorRatio"
    elif "crime" in text:
        canonical_task = "ViolentCrime"
        canonical_target = "ViolentCrime"
    elif ("build" in text and "height" in text) or "buildingheight" in text:
        canonical_task = "BuildHeight"
        canonical_target = "BuildHeight"

    if not isinstance(normalized_range, list) or len(normalized_range) != 2:
        normalized_range = [0.0, 9.9]

    return {
        "task_name": canonical_task,
        "target_field": canonical_target,
        "unit": unit,
        "normalized_range": normalized_range
    }


def parse_task_from_query(query, use_mock=False):
    if use_mock:
        return _canonicalize_task_spec({
            "task_name": "GDP",
            "target_field": "GDP",
            "unit": "PPP 2005 international dollars",
            "normalized_range": [0.0, 9.9]
        })

    prompt = build_task_parser_prompt(query)
    raw_output = LLM(prompt)

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    try:
        clean_text = extract_json(raw_output)
        data = json.loads(clean_text)
        return _canonicalize_task_spec(data)
    except Exception as e:
        print("Task parser error:", e)
        print("RAW:", raw_output)
        return _canonicalize_task_spec({
            "task_name": "GDP",
            "target_field": "GDP",
            "unit": None,
            "normalized_range": [0.0, 9.9]
        })



