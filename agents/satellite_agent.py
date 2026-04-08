import os

from core.schemas import Evidence
from core.llm_api import VLM
from core.task_guidance import get_task_guidance
import json
import re


def build_satellite_prompt(query, task_spec):
    task_name = task_spec["task_name"]
    target_field = str(task_spec.get("target_field", task_name)).lower()
    retrieved_guidance = get_task_guidance(target_field, "satellite")
    modality_guidance = """
- Treat overhead imagery as large-scale structural context.
- Density, roads, greenery, water, or mixed land use are not automatic positive evidence.
- If the image is weak or only broadly informative, stay near neutral.
"""

    prompt = f"""
You are an expert in interpreting satellite imagery for urban socioeconomic analysis.

Your task is to extract ONE structured evidence object from this satellite image for estimating {task_name}.

For this image:
- First judge whether it is informative, mixed, or weakly informative.
- Use only visible content.
- Use "unclear" when built form or land use is ambiguous.
- Positive implication only when built environment clearly supports above-average conditions.
- Negative implication when the image suggests sparse development, weak connectivity, or limited built-up structure.

Modality guidance:
{modality_guidance}

Retrieved task guidance:
{retrieved_guidance}

IMPORTANT:
- You must output exactly ONE JSON object.
- Do NOT output a list.
- Be descriptive and grounded in visible content.

Use this structure:
- observation
- scene_type
- spatial_layout:
  density / building_pattern / road_structure / land_use
- key_elements (2-4 items with type and description)
- local_variation
- coverage
- informativeness in [0,1]
- implication in [-1,1]
- confidence in [0,1]

Return exactly one JSON object in this format:

{{
  "observation": "...",
  "scene_type": "built_up_dominant | mixed | weakly_informative",
  "spatial_layout": {{
    "density": "...",
    "building_pattern": "...",
    "road_structure": "...",
    "land_use": "..."
  }},
  "key_elements": [
    {{"type": "...", "description": "..."}},
    {{"type": "...", "description": "..."}}
  ],
  "local_variation": "...",
  "coverage": "...",
  "informativeness": 0.0,
  "implication": 0.0,
  "confidence": 0.0
}}

scene_type must be one of:
- built_up_dominant
- mixed
- weakly_informative

Use low informativeness and near-neutral implication for weak images.
Output JSON only. Do not invent details.

User query:
{query}
"""
    return prompt


def extract_json(text):
    text = text.strip()

    code_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(0)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def parse_satellite_output(raw_output, image_path, task_spec, start_idx=0):
    parsed_evidences = []

    try:
        clean_text = extract_json(raw_output)
        data = json.loads(clean_text)

        if isinstance(data, dict):
            data = [data]

        for i, item in enumerate(data):
            implication = float(item.get("implication", 0.0))
            target_field = task_spec["target_field"]

            if str(target_field).lower() in ("houseprice", "house_price"):
                implication = 0.0

            evidence = Evidence(
                eid=f"E_sat_{start_idx + i}",
                modality="satellite",
                observation=item.get("observation"),
                implication={target_field: implication},
                confidence=float(item.get("confidence", 0.5)),
                source=os.path.relpath(image_path),
                spatial_layout=item.get("spatial_layout", {}),
                key_elements=item.get("key_elements", []),
                local_variation=item.get("local_variation"),
                coverage=item.get("coverage"),
                semantic_type="satellite_structured",
                scene_type=item.get("scene_type"),
                informativeness=item.get("informativeness"),
            )
            parsed_evidences.append(evidence)

    except Exception as e:
        print("Parse error:", e)
        print("RAW:", raw_output)
        return []

    return parsed_evidences


def satellite_agent(image_path, query, task_spec, use_mock=False):
    prompt = build_satellite_prompt(query, task_spec)
    raw_output = VLM([image_path], prompt)

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    evidences = parse_satellite_output(raw_output, image_path, task_spec, start_idx=0)
    return evidences or []


