#!/usr/bin/env python3
"""CLI pipeline to annotate AMEX screens for purpose/expectation with local Qwen-VL.

Example:
python -u annotate_amex_qwen.py \
  --qwen-model /path/to/Qwen3-VL-30B-A3B-Instruct \
  --anno-dir ./element_anno \
  --image-dir ./screenshots \
  --out-purpose ./purpose \
  --out-expectation ./expectation \
  --qwen-max-edge 1024 \
  --qwen-max-new-tokens 512 \
  --qwen-batch-size 1 \
  --qwen-use-fast-processor \
  --test --test-samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

SYSTEM_PROMPT = (
    "You are an expert mobile UI analyst. "
    "For each screen, infer (1) user purpose and (2) immediate expectation after interaction. "
    "Use only evidence from the image and structured annotation."
)

USER_PROMPT_TEMPLATE = """
You are given a mobile app screenshot with structured accessibility annotations.

Task:
Return STRICT JSON with exactly two keys:
{{
  "purpose": "one concise sentence for the user's likely high-level goal on this screen",
  "expectation": "one concise sentence for what the user expects to happen next if they take the primary action"
}}

Guidelines:
- Purpose = high-level intent for current screen context.
- Expectation = immediate outcome the user anticipates from the most salient/primary action.
- Be specific and grounded in provided metadata.
- No markdown. No extra keys.

Page caption:
{page_caption}

Key clickable elements (top {max_clickables}):
{clickable_text}

Key scrollable elements (top {max_scrollables}):
{scrollable_text}
""".strip()


@dataclass
class Sample:
    anno_path: Path
    image_path: Path
    page_caption: str
    clickable_elements: list[dict[str, Any]]
    scrollable_elements: list[dict[str, Any]]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def log_gpu_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        logging.info("[%s] CUDA not available.", tag)
        return
    parts = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        parts.append(f"GPU{i}: alloc={allocated:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB")
    logging.info("[%s] %s", tag, " | ".join(parts))


def resolve_input_device(model: AutoModelForImageTextToText) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for _, mapped in model.hf_device_map.items():
            mapped_str = str(mapped)
            if mapped_str.startswith("cuda"):
                return torch.device(mapped_str)
    return model.device


def resize_if_needed(img: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / float(longest)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def compact_element_list(elements: list[dict[str, Any]], limit: int) -> str:
    rows: list[str] = []
    for item in elements[:limit]:
        idx = item.get("idx", "NA")
        etype = item.get("type", "")
        functionality = item.get("functionality", "")
        xml_desc = item.get("xml_desc", [])
        text_hint = " | ".join(str(x) for x in xml_desc[:3]) if isinstance(xml_desc, list) else str(xml_desc)
        rows.append(f"- idx={idx}; type={etype}; text={text_hint}; functionality={functionality}")
    return "\n".join(rows) if rows else "- None"


def build_user_prompt(sample: Sample, max_clickables: int = 18, max_scrollables: int = 8) -> str:
    return USER_PROMPT_TEMPLATE.format(
        page_caption=sample.page_caption,
        max_clickables=max_clickables,
        clickable_text=compact_element_list(sample.clickable_elements, max_clickables),
        max_scrollables=max_scrollables,
        scrollable_text=compact_element_list(sample.scrollable_elements, max_scrollables),
    )


def load_samples(anno_dir: Path, image_dir: Path, test: bool, test_samples: int) -> list[Sample]:
    if not anno_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {anno_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    samples: list[Sample] = []
    for ap in sorted(anno_dir.glob("*.json")):
        try:
            data = json.loads(ap.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Skipping invalid JSON: %s (%s)", ap, exc)
            continue

        img_rel = data.get("image_path")
        if not img_rel:
            logging.warning("Skipping %s: missing image_path", ap)
            continue

        ip = image_dir / img_rel
        if not ip.exists():
            logging.warning("Skipping %s: missing image %s", ap, ip)
            continue

        samples.append(
            Sample(
                anno_path=ap,
                image_path=ip,
                page_caption=str(data.get("page_caption", "")),
                clickable_elements=list(data.get("clickable_elements", [])),
                scrollable_elements=list(data.get("scrollable_elements", [])),
            )
        )

    if test:
        samples = samples[: max(0, test_samples)]

    logging.info("Loaded %d samples (test=%s)", len(samples), test)
    return samples


def parse_model_json(text: str) -> dict[str, str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {
                "purpose": str(obj.get("purpose", "")).strip(),
                "expectation": str(obj.get("expectation", "")).strip(),
            }
    except json.JSONDecodeError:
        pass

    purpose = ""
    expectation = ""
    for line in text.splitlines():
        low = line.lower().strip()
        if low.startswith("purpose") and not purpose:
            purpose = line.split(":", 1)[-1].strip()
        elif low.startswith("expectation") and not expectation:
            expectation = line.split(":", 1)[-1].strip()
    return {"purpose": purpose, "expectation": expectation}


def save_prediction(sample: Sample, parsed: dict[str, str], raw_text: str, out_purpose: Path, out_expectation: Path) -> None:
    out_purpose.mkdir(parents=True, exist_ok=True)
    out_expectation.mkdir(parents=True, exist_ok=True)
    stem = sample.anno_path.stem

    (out_purpose / f"{stem}.json").write_text(
        json.dumps(
            {
                "source_annotation": str(sample.anno_path),
                "image_path": str(sample.image_path),
                "purpose": parsed.get("purpose", ""),
                "raw_response": raw_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_expectation / f"{stem}.json").write_text(
        json.dumps(
            {
                "source_annotation": str(sample.anno_path),
                "image_path": str(sample.image_path),
                "expectation": parsed.get("expectation", ""),
                "raw_response": raw_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    setup_logging()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_gpu_memory("startup")

    samples = load_samples(Path(args.anno_dir), Path(args.image_dir), args.test, args.test_samples)
    if not samples:
        logging.warning("No valid samples found. Exiting.")
        return

    processor = AutoProcessor.from_pretrained(
        args.qwen_model,
        trust_remote_code=True,
        use_fast=args.qwen_use_fast_processor,
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and hasattr(processor.tokenizer, "eos_token"):
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.qwen_model,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map=args.qwen_device_map,
    )
    model.eval()
    if hasattr(model, "generation_config"):
        # Disable KV cache by default for memory stability on near-full VRAM setups.
        model.generation_config.use_cache = args.qwen_use_cache
        torch_dtype=torch.float16,
        device_map=args.qwen_device_map,
    )
    model.eval()

    input_device = resolve_input_device(model)
    logging.info("Resolved model input device: %s", input_device)
    if hasattr(model, "hf_device_map"):
        logging.info("hf_device_map: %s", model.hf_device_map)

    log_gpu_memory("model_loaded")

    idx = 0
    while idx < len(samples):
        current_batch = min(args.qwen_batch_size, len(samples) - idx)
        while current_batch >= 1:
            batch = samples[idx : idx + current_batch]
            images: list[Image.Image] = []
            log_gpu_memory(f"before_batch_idx_{idx}_size_{current_batch}")
            try:
                prompts: list[str] = []
                for s in batch:
                    conversation = [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": build_user_prompt(s)},
                            ],
                        },
                    ]
                    prompts.append(processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
                    images.append(resize_if_needed(Image.open(s.image_path).convert("RGB"), args.qwen_max_edge))

                inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
                inputs = {k: v.to(input_device) if hasattr(v, "to") else v for k, v in inputs.items()}

                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": args.qwen_max_new_tokens,
                    "do_sample": args.qwen_temperature > 0,
                }
                if gen_kwargs["do_sample"]:
                    gen_kwargs["temperature"] = args.qwen_temperature
                    gen_kwargs["top_p"] = args.qwen_top_p

                with torch.inference_mode():
                    generated = model.generate(**inputs, **gen_kwargs)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.qwen_max_new_tokens,
                        do_sample=args.qwen_temperature > 0,
                        temperature=args.qwen_temperature,
                        top_p=args.qwen_top_p,
                    )

                in_lens = inputs["attention_mask"].sum(dim=-1).tolist()
                generated_only = [seq[int(in_len) :] for seq, in_len in zip(generated, in_lens)]
                decoded = processor.batch_decode(generated_only, skip_special_tokens=True)

                for sample, text in zip(batch, decoded):
                    parsed = parse_model_json(text)
                    save_prediction(sample, parsed, text, Path(args.out_purpose), Path(args.out_expectation))
                    logging.info(
                        "Saved -> %s | purpose=%s | expectation=%s",
                        sample.anno_path.name,
                        parsed.get("purpose", "")[:100],
                        parsed.get("expectation", "")[:100],
                    )

                idx += current_batch
                del inputs, generated
                break

            except RuntimeError as err:
                err_text = str(err).lower()
                if "out of memory" in err_text and current_batch > 1:
                    logging.warning("OOM at batch=%d. Retrying with half batch.", current_batch)
                    current_batch = max(1, current_batch // 2)
                    continue

                if "cuda driver error: invalid argument" in err_text and args.qwen_max_new_tokens > 64:
                    new_tokens = max(64, args.qwen_max_new_tokens // 2)
                    logging.warning(
                        "CUDA invalid argument during generate. Retrying sample with fewer max_new_tokens: %d -> %d",
                        args.qwen_max_new_tokens,
                        new_tokens,
                    )
                    args.qwen_max_new_tokens = new_tokens
                    continue

                raise
                if "out of memory" not in str(err).lower() or current_batch == 1:
                    raise
                logging.warning("OOM at batch=%d. Retrying with half batch.", current_batch)
                current_batch = max(1, current_batch // 2)
            finally:
                for im in images:
                    im.close()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_gpu_memory(f"after_batch_idx_{idx}_size_{current_batch}")

    logging.info("Done. Outputs: purpose=%s expectation=%s", args.out_purpose, args.out_expectation)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AMEX purpose/expectation annotation with local Qwen-VL")
    p.add_argument("--anno-dir", default="./element_anno")
    p.add_argument("--image-dir", default="./screenshots")
    p.add_argument("--out-purpose", default="./purpose")
    p.add_argument("--out-expectation", default="./expectation")

    p.add_argument("--qwen-model", required=True)
    p.add_argument("--qwen-device-map", default="auto")
    p.add_argument("--qwen-batch-size", type=int, default=1)
    p.add_argument("--qwen-max-edge", type=int, default=1024)
    p.add_argument("--qwen-max-new-tokens", type=int, default=512)
    p.add_argument("--qwen-temperature", type=float, default=0.0)
    p.add_argument("--qwen-top-p", type=float, default=1.0)
    p.add_argument("--qwen-use-fast-processor", action="store_true")
    p.add_argument("--qwen-use-cache", action="store_true", help="Enable generation KV cache (disabled by default for stability).")

    p.add_argument("--test", action="store_true")
    p.add_argument("--test-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.qwen_batch_size < 1:
        raise ValueError("--qwen-batch-size must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
