import json
import os
from PIL import Image
from torch.utils.data import Dataset


class AmexUIActionDataset:
    """
    Dataset processor for AMEX UI Action annotations.

    Expected JSON format (list of records):
    {
        "image":  "filename.png",
        "prefix": "<UI_ACTION> Click to view and book airport transfer services.",
        "suffix": "<loc_525><loc_425><loc_671><loc_492>"
    }

    - Coordinates in suffix are already in 0-1000 scale — passed through as-is.
    - 'click' is prepended to the suffix to match the existing ActionDataset suffix
      format: "click <loc_x1><loc_y1><loc_x2><loc_y2>"
    - The prefix already contains the <UI_ACTION> token — used directly.
    - image_dir: directory containing all image files.
    """

    def __init__(self, json_path: str, image_dir: str, max_samples: int = None):
        """
        Args:
            json_path:   Path to the annotations JSON file.
            image_dir:   Directory containing image files referenced by "image" field.
            max_samples: Optional cap on number of samples to load (applied after load).
        """
        self.json_path = json_path
        self.image_dir = image_dir
        self.max_samples = max_samples
        self.preprocessed_data = None
        self._load_and_preprocess()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_preprocess(self):
        """Load the JSON file and build the preprocessed sample list."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            print(f"[AmexUIActionDataset] ERROR reading {self.json_path}: {e}")
            raw_data = []

        total = len(raw_data)
        skipped = 0
        self.preprocessed_data = []

        for record in raw_data:
            image_name = record.get("image", "").strip()
            prefix = record.get("prefix", "").strip()
            suffix = record.get("suffix", "").strip()

            if not image_name or not prefix or not suffix:
                skipped += 1
                continue

            # Prepend 'click' to the coordinate suffix to match the existing
            # ActionDataset suffix style: "click <loc_x1><loc_y1><loc_x2><loc_y2>"
            if not suffix.lower().startswith("click"):
                suffix = f"click {suffix}"

            self.preprocessed_data.append({
                "image_id": image_name,
                "prefix":   prefix,
                "suffix":   suffix,
            })

        # Apply optional sample cap
        if self.max_samples is not None and self.max_samples > 0:
            self.preprocessed_data = self.preprocessed_data[: self.max_samples]

        loaded = len(self.preprocessed_data)
        print(
            f"[AmexUIActionDataset] JSON path : {self.json_path}\n"
            f"[AmexUIActionDataset] Image dir : {self.image_dir}\n"
            f"[AmexUIActionDataset] Total records  : {total:,}\n"
            f"[AmexUIActionDataset] Skipped (empty) : {skipped:,}\n"
            f"[AmexUIActionDataset] Loaded samples  : {loaded:,}"
            + (f" (capped from {total - skipped:,})" if self.max_samples and loaded < total - skipped else "")
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
            print(f"[AmexUIActionDataset] WARNING: Could not load image '{image_path}': {e}")
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

class FlorenceAmexUIActionDataset(Dataset):
    """PyTorch Dataset wrapper for AmexUIActionDataset — lazy image loading at collate time."""

    def __init__(self, amex_ui_dataset: AmexUIActionDataset):
        self.dataset = amex_ui_dataset
        self.data = amex_ui_dataset.getData()
        print(
            f"[FlorenceAmexUIActionDataset] Initialized with {len(self.data):,} samples"
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
