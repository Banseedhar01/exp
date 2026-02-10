# AMEX Purpose/Expectation Annotation (Qwen-VL)

This repository contains a CLI script (`annotate_amex_qwen.py`) to annotate AMEX UI data with:
- `purpose`: the high-level user goal on the current screen
- `expectation`: what the user expects to happen next after the primary action

It is designed for JSON annotations where one image can have many UI elements (often 15–20+ clickable/scrollable elements).

---

## Input data format

For each sample:
1. One JSON file in `element_anno/` (or `--anno-dir`)
2. One image in `screenshots/` (or `--image-dir`)

The JSON is expected to include fields like:
- `page_caption`
- `image_path`
- `clickable_elements` (list of UI elements with `idx`, `type`, `xml_desc`, `functionality`, `bbox`)
- `scrollable_elements`

---

## Basic processing flow

### 1) Load and validate sample pairs
- Script scans `--anno-dir` for `*.json`.
- For each JSON, it reads `image_path` and checks that the corresponding image exists in `--image-dir`.
- Invalid/missing pairs are skipped with warnings.
- If `--test` is enabled, it keeps only `--test-samples` entries.

### 2) Build a prompt per JSON
For each valid sample, the script prepares a structured prompt from:
- the `page_caption`
- top clickable elements (up to 18)
- top scrollable elements (up to 8)

So when one screen has 15–20+ elements, the model gets concise but rich context from those elements.

### 2.1) How each element is used in the prompts (System + User)

The script sends two roles to Qwen for every image:

#### System prompt (global behavior)
- Sets the model role as an expert mobile UI analyst.
- Instructs the model to infer:
  1. `purpose` (high-level user goal)
  2. `expectation` (immediate expected outcome after primary action)
- Enforces grounding in image + structured metadata.

#### User prompt (per sample content)
- Includes the current screen `page_caption`.
- Includes compacted clickable and scrollable element lists.
- For each element, the text line is built like:

```text
- idx=<idx>; type=<type>; text=<xml_desc snippets>; functionality=<functionality>
```

Where fields contribute as follows:
- `idx`: stable element reference ID.
- `type`: control type hint (icon/button/etc.).
- `xml_desc`: visible/accessibility text snippets (first few values).
- `functionality`: action semantics (what clicking/scrolling does).

This means when one JSON has 15–20+ elements, each element contributes a compact semantic line; the model uses these lines together with the image to infer the final two outputs.

### 3) Prepare multimodal batch
- For each batch (`--qwen-batch-size`), images are loaded and optionally resized (`--qwen-max-edge`) to reduce VRAM pressure.
- The script applies the Qwen chat template with image + text prompt.
- Inputs are moved to the resolved model input device (works with `device_map=auto` / multi-GPU).

### 4) Run Qwen generation
- Model: `AutoModelForImageTextToText`
- Processor: `AutoProcessor`
- Main generation controls:
  - `--qwen-max-new-tokens`
  - `--qwen-temperature`
  - `--qwen-top-p`

### 5) Decode and parse output
- Generated tokens are trimmed to remove input prompt tokens before decoding.
- Script expects strict JSON output with keys:
  - `purpose`
  - `expectation`
- If strict JSON parsing fails, fallback line-based parsing is used.

### 6) Save outputs per sample
For each input JSON `<name>.json`, script writes:
- `--out-purpose/<name>.json`
- `--out-expectation/<name>.json`

Each output includes source paths + extracted field + raw model response.

### 7) Memory and resilience
- GPU memory logs are printed at startup, after model load, and around each batch.
- On CUDA OOM, script automatically retries with half batch size until success (or batch size 1).

---

## Example command

```bash
python -u annotate_amex_qwen.py \
  --qwen-model /group-volume/SRIB-Bixby-Screen-AI/k.anup/Qwen-models/Qwen3-VL-30B-A3B-Instruct/ \
  --qwen-max-edge 1024 \
  --qwen-max-new-tokens 512 \
  --qwen-batch-size 1 \
  --qwen-use-fast-processor \
  --image-dir /group-volume/SRIB-Bixby-Screen-AI/k.anup/dataset/ScreenAI/ui_auto-split/1/screenshots \
  --anno-dir /group-volume/SRIB-Bixby-Screen-AI/k.anup/dataset/ScreenAI/ui_auto-split/1/element_anno \
  --out-purpose /group-volume/SRIB-Bixby-Screen-AI/k.anup/dataset/ScreenAI/ui_auto-split/1-processed/purpose \
  --out-expectation /group-volume/SRIB-Bixby-Screen-AI/k.anup/dataset/ScreenAI/ui_auto-split/1-processed/expectation \
  --test --test-samples 10
```

---

## Tips for 4x V100

- Start with `--qwen-batch-size 1`, then increase gradually.
- Keep `--qwen-max-edge` at 1024 (or lower) if VRAM is tight.
- Use `--qwen-temperature 0.0` for deterministic labels.

## Troubleshooting

- If startup seems stuck after `[startup]`, it is usually model shard loading for 30B checkpoints from storage; wait for `Loading checkpoint shards` and then `[model_loaded]`.
- If you hit `RuntimeError: CUDA driver error: invalid argument` during generation:
  - keep `--qwen-batch-size 1`
  - reduce `--qwen-max-new-tokens` (e.g., 512 -> 256 -> 128)
  - keep `--qwen-max-edge 1024` or lower
  - avoid sampling (`--qwen-temperature 0.0`) unless needed
- The script now retries invalid-argument generation once by reducing `max_new_tokens` automatically.
