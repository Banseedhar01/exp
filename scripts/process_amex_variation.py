"""
process_amex_variation.py
=========================
Process AMEX annotation JSON files from a directory and produce two output JSON files:
  - output_action.json   : one entry per functionality_variation  (UI_ACTION)
  - output_purpose.json  : one entry per purpose string           (UI_PURPOSE)

BBox normalisation
------------------
  x1n = round(x1 / W * 1000), clamped to [0, 999]
  same for y1, x2, y2
  Token format: <loc_XXX>  (zero-padded to 3 digits)

Usage
-----
  python process_amex_variation.py \\
      --json_dir  dir1 dir2 dir3 ...                       \\
      --image_dir <path to directory with source images>   \\
      --out_dir   <output directory>                       \\
      [--workers  8]

--json_dir accepts one or more directories; JSON files from all of them
are merged into a single processing run.

The image WxH is read from each PNG/JPG file on disk so that bboxes are
normalised correctly per image.  Images that cannot be opened are skipped
with a warning.
"""

import argparse
import json
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(v: int, lo: int = 0, hi: int = 999) -> int:
    return max(lo, min(hi, v))


def normalise_bbox(bbox: list, W: int, H: int) -> str:
    """Convert absolute [x1,y1,x2,y2] to <loc_...> token string."""
    x1, y1, x2, y2 = bbox
    x1n = clamp(round(x1 / W * 1000))
    y1n = clamp(round(y1 / H * 1000))
    x2n = clamp(round(x2 / W * 1000))
    y2n = clamp(round(y2 / H * 1000))
    return f"<loc_{x1n:03d}><loc_{y1n:03d}><loc_{x2n:03d}><loc_{y2n:03d}>"


def get_image_size(image_path: Path) -> Optional[tuple]:
    """Return (W, H) of an image, or None if it cannot be opened."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as exc:
        log.warning("Cannot open image %s: %s", image_path, exc)
        return None


def parse_variations(raw) -> list:
    """
    Normalise functionality_variations regardless of how it arrives:
      - list of strings  → strip each item
      - single string    → split on comma, strip each part
      - anything else    → return empty list
    """
    if isinstance(raw, list):
        return [v.strip() for v in raw if str(v).strip()]
    if isinstance(raw, str) and raw.strip():
        return [v.strip() for v in raw.split(",") if v.strip()]
    return []


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(json_path: Path, image_dir: Path):
    """
    Process one annotation JSON file.

    Returns
    -------
    action_rows : list[dict]   – UI_ACTION entries
    purpose_rows : list[dict]  – UI_PURPOSE entries
    skipped : int              – elements skipped (missing image / bad data)
    """
    action_rows: list = []
    purpose_rows: list = []
    skipped = 0

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        log.error("Failed to read %s: %s", json_path, exc)
        return action_rows, purpose_rows, skipped

    # Resolve image
    image_name: str = data.get("image_path", "")
    image_path = image_dir / image_name
    size = get_image_size(image_path)
    if size is None:
        log.warning("Skipping %s – image not found or unreadable (%s)", json_path.name, image_path)
        return action_rows, purpose_rows, skipped

    W, H = size

    for element in data.get("clickable_elements", []):
        bbox = element.get("bbox")
        if not bbox or len(bbox) != 4:
            skipped += 1
            continue

        loc_token = normalise_bbox(bbox, W, H)
        image_name_out = image_name  # keep original filename

        # ── UI_ACTION (functionality_variations) ─────────────────────────
        # Supports both a list and a comma-separated string
        raw_variations = element.get("functionality_variations", [])
        variations = parse_variations(raw_variations)
        for variation in variations:
            if not variation:
                skipped += 1
                continue
            action_rows.append({
                "image": image_name_out,
                "prefix": f"<UI_ACTION> {variation}",
                "suffix": loc_token,
            })

        # ── UI_PURPOSE (purpose) ─────────────────────────────────────────
        purpose = element.get("purpose", "").strip()
        if purpose:
            purpose_rows.append({
                "image": image_name_out,
                "prefix": f"<UI_PURPOSE> {loc_token}",
                "suffix": purpose,
            })
        else:
            skipped += 1

    return action_rows, purpose_rows, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process AMEX annotation JSONs → output_action.json + output_purpose.json"
    )
    parser.add_argument(
        "--json_dir", required=True, nargs="+", metavar="DIR",
        help="One or more directories containing annotation JSON files"
    )
    parser.add_argument("--image_dir", required=True, help="Directory containing source images")
    parser.add_argument("--out_dir",   required=True, help="Output directory")
    parser.add_argument("--workers",   type=int, default=8, help="Thread-pool size (default: 8)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir   = Path(args.out_dir)

    if not image_dir.is_dir():
        log.error("--image_dir is not a directory: %s", image_dir)
        sys.exit(1)

    # Validate all json_dirs and collect files
    json_dirs = []
    for d in args.json_dir:
        p = Path(d)
        if not p.is_dir():
            log.error("Not a directory (skipping): %s", p)
        else:
            json_dirs.append(p)

    if not json_dirs:
        log.error("No valid JSON directories provided.")
        sys.exit(1)

    # Gather JSON files from all directories (preserve per-dir counts for stats)
    json_files = []
    dir_counts: dict = {}
    for d in json_dirs:
        files = sorted(d.glob("*.json"))
        dir_counts[str(d)] = len(files)
        json_files.extend(files)
        log.info("  %d JSON files in %s", len(files), d)

    if not json_files:
        log.warning("No JSON files found across all provided directories.")
        sys.exit(0)

    log.info("Total JSON files to process: %d across %d director(y/ies)",
             len(json_files), len(json_dirs))
    log.info("Using %d worker threads", args.workers)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_action: list = []
    all_purpose: list = []
    total_files_ok = 0
    total_skipped  = 0

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for jf in json_files:
            fut = pool.submit(process_file, jf, image_dir)
            futures[fut] = jf

        with tqdm(total=len(json_files), desc="Processing JSONs", unit="file") as pbar:
            for fut in as_completed(futures):
                jf = futures[fut]
                try:
                    a_rows, p_rows, skipped = fut.result()
                    all_action.extend(a_rows)
                    all_purpose.extend(p_rows)
                    total_skipped  += skipped
                    total_files_ok += 1
                except Exception as exc:
                    log.error("Unexpected error processing %s: %s", jf.name, exc)
                finally:
                    pbar.update(1)

    # Write outputs
    action_out  = out_dir / "output_action.json"
    purpose_out = out_dir / "output_purpose.json"

    with open(action_out,  "w", encoding="utf-8") as fh:
        json.dump(all_action,  fh, indent=2, ensure_ascii=False)
    with open(purpose_out, "w", encoding="utf-8") as fh:
        json.dump(all_purpose, fh, indent=2, ensure_ascii=False)

    # ── Statistics ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Processing complete")
    print("=" * 60)
    print(f"  Input directories          : {len(json_dirs)}")
    for d, cnt in dir_counts.items():
        print(f"    {cnt:>6} files  ←  {d}")
    print(f"  Total JSON files found     : {len(json_files)}")
    print(f"  Files processed OK         : {total_files_ok}")
    print(f"  Files failed               : {len(json_files) - total_files_ok}")
    print(f"  UI_ACTION  samples         : {len(all_action)}")
    print(f"  UI_PURPOSE samples         : {len(all_purpose)}")
    print(f"  Elements skipped (bad data): {total_skipped}")
    print(f"  Output → {action_out}")
    print(f"  Output → {purpose_out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
