import json
import os
from PIL import Image
from torch.utils.data import Dataset


class VQADataset:
    """
    Dataset processor for Visual Question Answering (VQA) annotations.

    Expected JSON format (list of records):
    {
        "image":  "img_0077571.png",
        "prefix": "<VQA> What is the age of Japtko?",
        "suffix": "Japtko is 29 years old.<loc_373><loc_211><loc_413><loc_238>"
    }

    - prefix must begin with the <VQA> task token — used directly.
    - suffix is a free-form answer, optionally followed by <loc_*> coordinates.
    - image_dir: directory containing all image files.
    """

    def __init__(self, json_path: str, image_dir: str, max_samples: int = None):
        """
        Args:
            json_path:   Path to the annotations JSON file (list of records).
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
            print(f"[VQADataset] ERROR reading {self.json_path}: {e}")
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
            f"[VQADataset] JSON path : {self.json_path}\n"
            f"[VQADataset] Image dir : {self.image_dir}\n"
            f"[VQADataset] Total records  : {total:,}\n"
            f"[VQADataset] Skipped (empty) : {skipped:,}\n"
            f"[VQADataset] Loaded samples  : {loaded:,}"
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
            print(f"[VQADataset] WARNING: Could not load image '{image_path}': {e}")
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

class FlorenceVQADataset(Dataset):
    """PyTorch Dataset wrapper for VQADataset — lazy image loading at collate time."""

    def __init__(self, vqa_dataset: VQADataset):
        self.dataset = vqa_dataset
        self.data = vqa_dataset.getData()
        print(
            f"[FlorenceVQADataset] Initialized with {len(self.data):,} samples"
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
