import json
import os
from pathlib import Path

from main_update import run_single_case


ROOT = Path(__file__).resolve().parent
SAMPLE_JSON = ROOT / "sample_case" / "sample.json"


def main():
    with SAMPLE_JSON.open("r", encoding="utf-8") as f:
        sample = json.load(f)

    os.chdir(ROOT)
    sample_dir = SAMPLE_JSON.parent
    sat_image = str(Path("sample_case") / sample["sat_image"])
    street_images = [str(Path("sample_case") / p) for p in sample["street_images"]]
    output_dir = sample.get("output_dir", "outputs_sample")

    result = run_single_case(
        original_task_name=sample.get("task_name", "sample_task"),
        case_id=sample["case_id"],
        sat_image=sat_image,
        street_images=street_images,
        query=sample["query"],
        use_mock=False,
        output_dir=output_dir,
        use_verification=True,
        use_refinement=True,
        setting_name="sample_run",
        use_contradiction_handling=True,
        use_reflection=True,
        use_balanced_aggregation=True,
    )

    print("Sample run completed.")
    print(f"Output directory: {output_dir}")
    print(f"Case ID: {result['case_id']}")
    print(f"Final estimate: {result['final_report'].get('final_estimate')}")


if __name__ == "__main__":
    main()
