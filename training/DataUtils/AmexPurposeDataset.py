import json
import os
from PIL import Image
from torch.utils.data import Dataset


class AmexPurposeDataset:
    """
    Dataset processor for AMEX UI Purpose annotations.

    Expected JSON format (list of records):
    {
        "image":  "filename.png",
        "prefix": "<UI_PURPOSE><loc_025><loc_949><loc_975><loc_990>",
        "suffix": "Add to cart button, a primary call-to-action for purchasing the product"
    }

    - The prefix already contains the <UI_PURPOSE> token + bounding box locs — used directly.
    - The suffix is a free-text purpose description — used as-is.
    - image_dir: directory containing all image files.
    - json_path: a single JSON file path (str) OR a list of paths for multi-file input.
    """

    def __init__(self, json_path, image_dir: str, max_samples: int = None):
        """
        Args:
            json_path:   Path to annotations JSON file, or a list of such paths.
            image_dir:   Directory containing image files referenced by "image" field.
            max_samples: Optional cap on total number of samples (applied after merging all files).
        """
        # Normalise to list so the rest of the code is uniform
        self.json_paths = [json_path] if isinstance(json_path, str) else list(json_path)
        self.image_dir = image_dir
        self.max_samples = max_samples
        self.preprocessed_data = None
        self._load_and_preprocess()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_preprocess(self):
        """Load all JSON files and merge into a single preprocessed sample list."""
        self.preprocessed_data = []
        grand_total = 0
        grand_skipped = 0

        for json_path in self.json_paths:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as e:
                print(f"[AmexPurposeDataset] ERROR reading {json_path}: {e}")
                raw_data = []

            file_total = len(raw_data)
            file_skipped = 0
            file_samples = []

            for record in raw_data:
                image_name = record.get("image", "").strip()
                prefix = record.get("prefix", "").strip()
                suffix = record.get("suffix", "").strip()

                if not image_name or not prefix or not suffix:
                    file_skipped += 1
                    continue

                # Prefix already contains <UI_PURPOSE> + loc tokens — use directly.
                # Suffix is a free-text purpose description — use as-is.
                file_samples.append({
                    "image_id": image_name,
                    "prefix":   prefix,
                    "suffix":   suffix,
                })

            grand_total += file_total
            grand_skipped += file_skipped
            self.preprocessed_data.extend(file_samples)
            print(
                f"[AmexPurposeDataset] Loaded '{json_path}': "
                f"{len(file_samples):,} samples "
                f"(skipped {file_skipped:,} / {file_total:,})"
            )

        # Apply optional sample cap across the merged dataset
        if self.max_samples is not None and self.max_samples > 0:
            self.preprocessed_data = self.preprocessed_data[: self.max_samples]

        loaded = len(self.preprocessed_data)
        print(
            f"[AmexPurposeDataset] --- Summary ---\n"
            f"[AmexPurposeDataset] JSON files  : {len(self.json_paths)}\n"
            f"[AmexPurposeDataset] Image dir   : {self.image_dir}\n"
            f"[AmexPurposeDataset] Total records  : {grand_total:,}\n"
            f"[AmexPurposeDataset] Skipped (empty) : {grand_skipped:,}\n"
            f"[AmexPurposeDataset] Loaded samples  : {loaded:,}"
            + (f" (capped from {grand_total - grand_skipped:,})" if self.max_samples and loaded < grand_total - grand_skipped else "")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image_name: str) -> Image.Image:
        """Load an image from the dataset's image_dir."""
        image_path = os.path.join(self.image_dir, image_name)
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[AmexPurposeDataset] WARNING: Could not load image '{image_path}': {e}")
            return Image.new("RGB", (224, 224), color="white")

    def getData(self):
        """Return the list of preprocessed sample dicts."""
        return self.preprocessed_data

    def getItem(self, index: int = 0):
        """Return a single item by index, or None if out of range."""
        if not self.preprocessed_data or index >= len(self.preprocessed_data):
            return None
        return self.preprocessed_data[index]

    def __len__(self):
        return len(self.preprocessed_data) if self.preprocessed_data else 0


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class FlorenceAmexPurposeDataset(Dataset):
    """PyTorch Dataset wrapper for AmexPurposeDataset — lazy image loading at collate time."""

    def __init__(self, amex_purpose_dataset: AmexPurposeDataset):
        self.dataset = amex_purpose_dataset
        self.data = amex_purpose_dataset.getData()
        print(
            f"[FlorenceAmexPurposeDataset] Initialized with {len(self.data):,} samples"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Returns (prefix, suffix, image_id, image_loader_fn)
        return (
            item["prefix"],
            item["suffix"],
            item["image_id"],
            self.dataset.load_image,
        )
