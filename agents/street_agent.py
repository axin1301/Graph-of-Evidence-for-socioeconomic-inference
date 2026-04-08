import json
import os
import re

from core.schemas import Evidence
from core.llm_api import VLM
from core.task_guidance import get_task_guidance


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


def build_street_prompt(query, task_spec, num_images=1):
    task_name = task_spec["task_name"]
    target_field = str(task_spec.get("target_field", task_name)).lower()
    retrieved_guidance = get_task_guidance(target_field, "street")
    modality_guidance = """
- Focus on visible built-environment quality, activity, accessibility, maintenance, and amenity signals.
- Roads, buildings, traffic, or vehicles alone are not enough for positive implication.
- If an image is weak, partial, or mixed, stay near neutral.
"""

    prompt = f"""
You are an expert in interpreting street-view imagery for urban socioeconomic analysis.

Your task is to extract structured evidence objects from {num_images} street-view image(s) for estimating {task_name}.

For each image:
- Decide whether it is informative, mixed, or weakly informative.
- Use only visible content.
- Use "unclear" when evidence is partial or ambiguous.
- Positive implication only when the scene clearly supports above-average conditions.
- Negative implication when the scene suggests weak street quality, low activity, poor maintenance, weak accessibility, or limited amenities.

Modality guidance:
{modality_guidance}

Retrieved task guidance:
{retrieved_guidance}

IMPORTANT:
- You must output exactly one JSON object per input image.
- Return a JSON list with {num_images} objects, one for each image, in input order.
- Each object must include an "image_index" field starting from 0.
- Do not merge multiple images into one summary.
- Be descriptive and grounded in visible content.

For each image, describe the scene using the following structure:
- observation
- scene_type
- streetscape:
  building_condition / street_activity / commercial_presence / greenery / transport_signals
- key_elements (2-4 items with type and description)
- local_variation
- coverage
- informativeness in [0,1]
- implication in [-1,1]
- confidence in [0,1]

Return exactly one JSON list in this format:

[
  {{
    "image_index": 0,
    "observation": "...",
    "scene_type": "built_environment_visible | mixed | weakly_informative",
    "streetscape": {{
      "building_condition": "...",
      "street_activity": "...",
      "commercial_presence": "...",
      "greenery": "...",
      "transport_signals": "..."
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
]

scene_type must be one of:
- built_environment_visible
- mixed
- weakly_informative

Use low informativeness and near-neutral implication for weak images.
Output JSON only. Do not invent details.

User query:
{query}
"""
    return prompt


def parse_street_output(raw_output, image_paths, task_spec, start_idx=0):
    parsed_evidences = []

    try:
        clean_text = extract_json(raw_output)
        data = json.loads(clean_text)

        if isinstance(data, dict):
            data = [data]

        for i, item in enumerate(data):
            image_index = item.get("image_index", i)
            try:
                image_index = int(image_index)
            except Exception:
                image_index = i
            if image_index < 0 or image_index >= len(image_paths):
                image_index = i if i < len(image_paths) else len(image_paths) - 1

            evidence = Evidence(
                eid=f"E_st_{start_idx + i}",
                modality="street",
                observation=item.get("observation"),
                implication={task_spec["target_field"]: float(item.get("implication", 0.0))},
                confidence=float(item.get("confidence", 0.5)),
                source=os.path.relpath(image_paths[image_index]),
                spatial_layout=item.get("streetscape", {}),
                key_elements=item.get("key_elements", []),
                local_variation=item.get("local_variation"),
                coverage=item.get("coverage"),
                semantic_type="street_structured",
                scene_type=item.get("scene_type"),
                informativeness=item.get("informativeness"),
            )
            parsed_evidences.append(evidence)

    except Exception as e:
        print("Street parse error:", e)
        print("RAW:", raw_output)
        return []

    return parsed_evidences


def street_agent(image_paths, query, task_spec, use_mock=False):
    all_evidences = []
    current_idx = 0

    for image_path in image_paths:
        prompt = build_street_prompt(query, task_spec, num_images=1)
        raw_output = VLM([image_path], prompt)

        if isinstance(raw_output, tuple):
            raw_output = raw_output[0]

        evidences = parse_street_output(raw_output, [image_path], task_spec, start_idx=current_idx)
        all_evidences.extend(evidences or [])
        current_idx += len(evidences or [])

    return all_evidences


