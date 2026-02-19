"""
eval_florence.py — Evaluation script for fine-tuned Florence-2 checkpoints.

Supports two dataset formats:
  1. Parquet (ScreenSpot format): --dataset-format parquet
     Each row has: image (bytes), bbox (normalized [0,1]), instruction, file_name
     Prefix: "<COMMAND> <instruction>"  or  "<UI_ACTION> <instruction>"

  2. JSON directory (our training format): --dataset-format json
     Each JSON file has: image (filename), prefix, suffix
     Evaluates whether predicted center falls within GT bbox from suffix.

Usage:
  # ScreenSpot parquet
  python eval_florence.py \\
    --model-path ./florence2_finetuned_final_epoch_9 \\
    --dataset-path ../datasets/screenspot/test-00000-of-00003.parquet \\
    --dataset-format parquet \\
    --task-token "<COMMAND>" \\
    --output-dir ./results/run1

  # JSON directory (COMMAND format)
  python eval_florence.py \\
    --model-path ./florence2_finetuned_final_epoch_9 \\
    --dataset-path ../datasets/command_data/ \\
    --dataset-format json \\
    --image-dir ../datasets/command_images/ \\
    --output-dir ./results/run1

  # Limit to first N samples
  python eval_florence.py --model-path ./ckpt --dataset-path data.parquet --max-samples 100
"""

import os
import re
import io
import json
import math
import glob
import logging
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Florence-2 Evaluation Script")
    parser.add_argument("--model-path",      type=str, required=True,
                        help="Path to saved Florence-2 checkpoint directory")
    parser.add_argument("--dataset-path",    type=str, required=True,
                        help="Path to .parquet file or directory of JSON files")
    parser.add_argument("--dataset-format",  type=str, default="parquet",
                        choices=["parquet", "json"],
                        help="Dataset format: 'parquet' (ScreenSpot) or 'json' (training format)")
    parser.add_argument("--image-dir",       type=str, default=None,
                        help="Image directory (required for json format)")
    parser.add_argument("--task-token",      type=str, default="<COMMAND>",
                        help="Task token prefix for parquet format (default: '<COMMAND>')")
    parser.add_argument("--image-size",      type=int, default=768,
                        help="Image size used during training (default: 768)")
    parser.add_argument("--max-samples",     type=int, default=None,
                        help="Limit evaluation to first N samples")
    parser.add_argument("--output-dir",      type=str, default="./eval_results",
                        help="Directory to save results, logs, and failure images")
    parser.add_argument("--save-images",     action="store_true",
                        help="Save annotated images for success and failure cases")
    parser.add_argument("--device",          type=str, default=None,
                        help="Device to use: 'cuda' or 'cpu' (auto-detected if not set)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def parse_loc_tokens(text: str):
    """Extract all <loc_N> values from model output → list of ints."""
    return [int(x) for x in re.findall(r'<loc_(\d+)>', text)]


def loc_to_pixel(val: int, dimension: int) -> int:
    """Convert a Florence loc-token value [0,999] to pixel coordinate."""
    return int(val / 999 * dimension)


def loc_bbox_to_pixels(locs, orig_w: int, orig_h: int):
    """
    Convert 4 loc-token values to pixel bbox on the original image.
    Returns (x1, y1, x2, y2) in original image pixel space.
    """
    if len(locs) < 4:
        return None
    x1 = loc_to_pixel(locs[0], orig_w)
    y1 = loc_to_pixel(locs[1], orig_h)
    x2 = loc_to_pixel(locs[2], orig_w)
    y2 = loc_to_pixel(locs[3], orig_h)
    return x1, y1, x2, y2


def loc_point_to_pixels(locs, orig_w: int, orig_h: int):
    """Convert 2 loc-token values to a click point in original image space."""
    if len(locs) < 2:
        return None
    cx = loc_to_pixel(locs[0], orig_w)
    cy = loc_to_pixel(locs[1], orig_h)
    return cx, cy


def center_inside_bbox(pred_bbox, gt_bbox) -> bool:
    """
    Check if the CENTER of pred_bbox falls inside gt_bbox.
    Both in the same coordinate space (original image pixels).
    """
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    cx = (px1 + px2) / 2.0
    cy = (py1 + py2) / 2.0

    return gx1 <= cx <= gx2 and gy1 <= cy <= gy2


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def draw_boxes(image: Image.Image, pred_bbox, gt_bbox) -> Image.Image:
    """Draw predicted (red) and GT (green) bboxes on image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if pred_bbox:
        draw.rectangle(list(pred_bbox), outline="red", width=4)
    if gt_bbox:
        draw.rectangle(list(gt_bbox), outline="green", width=4)
    return out


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

class ParquetDataset:
    """
    Loads ScreenSpot-style parquet file.

    Supported column layouts:
      Layout A (nested):  image = {"bytes": ..., "path": ...}
      Layout B (flat):    image.bytes = bytes,  image.path = str   ← this dataset
    Columns: file_name, bbox ([x1,y1,x2,y2] normalized 0-1),
             instruction, data_type, data_source, image.bytes, image.path
    """
    def __init__(self, path: str, task_token: str, image_size: int):
        import pandas as pd
        self.df = pd.read_parquet(path)
        self.task_token = task_token
        self.image_size = image_size
        # Detect column layout
        cols = set(self.df.columns)
        if "image.bytes" in cols:
            self._img_col = "image.bytes"       # flat layout (your dataset)
        elif "image" in cols:
            self._img_col = "image"             # nested layout
        else:
            raise ValueError(f"Cannot find image column. Available: {cols}")

    def __len__(self):
        return len(self.df)

    def _get_image_bytes(self, row) -> bytes:
        """Extract raw image bytes regardless of column layout."""
        val = row[self._img_col]
        if val is None:
            raise ValueError("image bytes is None (missing image data in this row)")
        if isinstance(val, dict):
            b = val.get("bytes")
            if b is None:
                raise ValueError("image dict has no 'bytes' key or it is None")
            return b
        return val   # flat layout — already bytes

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = self._get_image_bytes(row)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        # GT bbox: normalized [0,1] → pixel in original image space
        b = row["bbox"]
        gt_bbox = (
            b[0] * orig_w,
            b[1] * orig_h,
            b[2] * orig_w,
            b[3] * orig_h,
        )

        # Resize for model input
        img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        prompt = f"{self.task_token} {row['instruction']}".strip()

        # File name — prefer image.path then file_name
        file_name = str(row.get("image.path", row.get("file_name", f"sample_{idx}")))

        return {
            "prompt":       prompt,
            "image":        img_resized,
            "image_orig":   img,
            "gt_bbox":      gt_bbox,      # in original pixel space
            "orig_w":       orig_w,
            "orig_h":       orig_h,
            "file_name":    file_name,
            "idx":          idx,
        }



class JSONDataset:
    """
    Loads our training JSON format from a directory of .json files.
    Each JSON: {prefix, suffix, image}
    GT is parsed from the suffix loc-tokens (for COMMAND: single bbox or point).
    """
    def __init__(self, data_dir: str, image_dir: str, image_size: int):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if not self.files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")
        self.image_dir = image_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "r", encoding="utf-8") as f:
            item = json.load(f)

        img_path = os.path.join(self.image_dir, item["image"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Parse GT from suffix loc-tokens
        locs = parse_loc_tokens(item.get("suffix", ""))
        if len(locs) >= 4:
            gt_bbox = loc_bbox_to_pixels(locs[:4], orig_w, orig_h)
        elif len(locs) >= 2:
            cx, cy = loc_point_to_pixels(locs[:2], orig_w, orig_h)
            # Treat point as tiny 1×1 bbox for center-in-bbox check
            gt_bbox = (cx, cy, cx + 1, cy + 1)
        else:
            gt_bbox = None

        img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        return {
            "prompt":       item["prefix"],
            "image":        img_resized,
            "image_orig":   img,
            "gt_bbox":      gt_bbox,
            "orig_w":       orig_w,
            "orig_h":       orig_h,
            "file_name":    os.path.basename(self.files[idx]),
            "idx":          idx,
        }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, sample: dict, device, args) -> str:
    """Run model inference on a single sample, return raw generated text."""
    inputs = processor(
        text=sample["prompt"],
        images=[sample["image"]],
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=200,
            num_beams=3,
            do_sample=False,
            early_stopping=True
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args, logger):
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    model.eval()
    logger.info("Model loaded.")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_path} (format={args.dataset_format})")
    if args.dataset_format == "parquet":
        dataset = ParquetDataset(args.dataset_path, args.task_token, args.image_size)
    else:
        if not args.image_dir:
            raise ValueError("--image-dir is required when --dataset-format=json")
        dataset = JSONDataset(args.dataset_path, args.image_dir, args.image_size)

    total = len(dataset)
    if args.max_samples:
        total = min(total, args.max_samples)
    logger.info(f"Evaluating {total} samples")

    # Output dirs
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "failure"), exist_ok=True)

    # Eval loop
    success = 0
    failure = 0
    skipped = 0
    results = []

    for i in tqdm(range(total), desc="Evaluating"):
        try:
            sample = dataset[i]
        except Exception as e:
            logger.warning(f"[{i}] Failed to load sample: {e}")
            skipped += 1
            continue

        if sample["gt_bbox"] is None:
            logger.warning(f"[{i}] No GT bbox in suffix, skipping.")
            skipped += 1
            continue

        # Inference
        try:
            generated_text = run_inference(model, processor, sample, device, args)
        except Exception as e:
            logger.error(f"[{i}] Inference failed: {e}")
            skipped += 1
            continue

        # Parse predicted bbox
        locs = parse_loc_tokens(generated_text)
        if len(locs) < 2:
            logger.warning(f"[{i}] No loc tokens in output: {generated_text!r}")
            skipped += 1
            results.append({"idx": i, "file": sample["file_name"],
                            "status": "skipped", "output": generated_text})
            continue

        # Convert predicted locs → original pixel space
        orig_w, orig_h = sample["orig_w"], sample["orig_h"]
        if len(locs) >= 4:
            pred_bbox = loc_bbox_to_pixels(locs[:4], orig_w, orig_h)
        else:
            # Point prediction — expand to small bbox for center check
            cx, cy = loc_point_to_pixels(locs[:2], orig_w, orig_h)
            pred_bbox = (cx, cy, cx + 1, cy + 1)

        # Evaluate
        hit = center_inside_bbox(pred_bbox, sample["gt_bbox"])
        if hit:
            success += 1
            status = "success"
        else:
            failure += 1
            status = "failure"

        # Save annotated images
        if args.save_images:
            try:
                annotated = draw_boxes(sample["image_orig"], pred_bbox, sample["gt_bbox"])
                safe_name = re.sub(r'[<>:"/\\|?*]', '_', sample["file_name"])
                annotated.save(os.path.join(args.output_dir, status, f"{i}_{safe_name}.png"))
            except Exception as e:
                logger.warning(f"[{i}] Image save failed: {e}")

        results.append({
            "idx":          i,
            "file":         sample["file_name"],
            "prompt":       sample["prompt"],
            "output":       generated_text,
            "pred_bbox":    list(pred_bbox),
            "gt_bbox":      list(sample["gt_bbox"]),
            "status":       status,
        })

        # Live progress
        done = success + failure
        acc = success / done * 100 if done > 0 else 0.0
        tqdm.write(f"[{i:>5}] {status.upper():7s} | acc={acc:.1f}% "
                   f"({success}/{done}) | {sample['file_name']}")

    # Final summary
    done = success + failure
    accuracy = success / done * 100 if done > 0 else 0.0
    logger.info("=" * 60)
    logger.info(f"RESULTS")
    logger.info(f"  Total evaluated : {done}")
    logger.info(f"  Skipped         : {skipped}")
    logger.info(f"  Success         : {success}")
    logger.info(f"  Failure         : {failure}")
    logger.info(f"  Accuracy        : {accuracy:.2f}%")
    logger.info("=" * 60)

    # Save full results JSON
    summary = {
        "model_path":    args.model_path,
        "dataset_path":  args.dataset_path,
        "dataset_format":args.dataset_format,
        "task_token":    args.task_token,
        "total":         done,
        "skipped":       skipped,
        "success":       success,
        "failure":       failure,
        "accuracy_pct":  round(accuracy, 4),
        "timestamp":     datetime.now().isoformat(),
        "samples":       results,
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to: {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logging(args.output_dir)
    logger.info("Florence-2 Evaluation")
    logger.info(f"Model      : {args.model_path}")
    logger.info(f"Dataset    : {args.dataset_path} ({args.dataset_format})")
    logger.info(f"Output dir : {args.output_dir}")
    evaluate(args, logger)
