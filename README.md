# Code for Submission

This folder contains a cleaned subset of the Graph-of-Evidence (GoE) codebase for anonymous submission.

## Directory Structure

- `main_update.py`: main entry point for running the GoE pipeline on one case.
- `agents/`: evidence grounding, claim generation, verification, refinement, reflection, and final report modules.
- `graph/`: GoE construction, augmentation, edge weighting, factor definitions, and final solver.
- `core/`: schemas, task parsing, task guidance, API wrapper, and I/O utilities.

## Required Environment`r`n`r`nBefore running, add your OpenRouter API key in `core/llm_api.py` by replacing the `YOUR_OPENROUTER_API_KEY` placeholder.`r`n`r`nAlternatively, you may provide the key through the `OPENROUTER_API_KEY` environment variable.`r`n
Optional environment variables:

- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_VLM_MODEL` (default: `qwen/qwen2.5-vl-7b-instruct`)
- `OPENROUTER_LLM_MODEL` (default: same as `OPENROUTER_VLM_MODEL`)
- `OPENROUTER_MAX_TOKENS` (default: `4096`)
- `OPENROUTER_TIMEOUT_SECONDS` (default: `180`)
- `MAX_STREET_IMAGES` (default: `10`)
- `UPDATE_SHARED_EVIDENCE_ROOTS` (optional comma-separated cache roots)

## Minimal Dependencies`r`n`r`n- Python 3.10+`r`n- `openai``r`n`r`nIf provided, `requirements.txt` corresponds to the Python environment used to run this code.`r`n
## Usage

Import and call `run_single_case(...)` from `main_update.py`.
The pipeline performs:

1. task parsing
2. satellite and street-view evidence grounding
3. initial claim generation
4. verification, refinement, and optional reflection
5. base and augmented Graph-of-Evidence construction
6. edge weighting and final claim solving
7. final report generation

## Notes

- This submission version removes hard-coded credentials and non-essential local setup.
- API credentials must be provided through environment variables.
- Cached evidence and outputs are written by `main_update.py` to the configured output directory.

## Run a Sample Case

Before running, add your OpenRouter API key in `core/llm_api.py` by replacing `YOUR_OPENROUTER_API_KEY`.

Run the sample pipeline:

```bash
python run_sample.py
```

The outputs will be written to `outputs_sample/` inside this folder.


