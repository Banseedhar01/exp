"""
Command Dataset Processor for Florence-2
Handles dataset format:
  {"image": "img.png", "prefix": "<COMMAND> Opens the details page for 'X'.", "suffix": "<loc_24><loc_440><loc_365><loc_597>"}

Supports both single JSON file and directory of individual JSON files.
"""

import json
import logging
import os
import glob
from typing import Tuple, Dict, Any
from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class CommandDatasetProcessor(BaseDatasetProcessor):
    """
    Processor for Command/Action format datasets.

    Expected JSON format:
    {
        "image": "image_name.png",
        "prefix": "<COMMAND> Action description here.",
        "suffix": "<loc_x1><loc_y1><loc_x2><loc_y2>"
    }
    """

    def __init__(self, data_path: str, image_dir: str, image_size: int = 768, resize_images: bool = True):
        """
        Args:
            data_path: Path to JSON file OR directory containing individual JSON files
            image_dir: Directory containing images
            image_size: Target size for image resizing
            resize_images: Whether to resize images
        """
        super().__init__(data_path, image_dir, image_size, resize_images)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load from file or directory."""
        logger.info(f"Loading COMMAND format data from {self.data_path}")
        try:
            if os.path.isdir(self.data_path):
                self._load_from_directory()
            else:
                self._load_from_file()
            logger.info(f"Loaded {len(self.data)} valid COMMAND samples")
        except Exception as e:
            logger.error(f"Error loading COMMAND data: {e}")
            raise

    def _load_from_file(self):
        """Load from a single JSON file (list or single object)."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        for idx, item in enumerate(raw_data):
            if self._validate_item(item, idx):
                self.data.append(item)

    def _load_from_directory(self):
        """Load from a directory of individual JSON files."""
        json_files = sorted(glob.glob(os.path.join(self.data_path, "*.json")))
        if not json_files:
            logger.warning(f"No JSON files found in {self.data_path}")
            return
        logger.info(f"Found {len(json_files)} JSON files")
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                if self._validate_item(item, len(self.data)):
                    self.data.append(item)
            except Exception as e:
                logger.warning(f"Skipping {json_file}: {e}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_item(self, item: Dict, idx: int) -> bool:
        """Validate a single COMMAND format item."""
        for field in ['image', 'prefix', 'suffix']:
            if field not in item:
                logger.warning(f"Item {idx} missing field: {field}")
                return False

        if not isinstance(item['prefix'], str) or not item['prefix'].strip():
            logger.warning(f"Item {idx} has empty prefix")
            return False

        if '<loc_' not in item['suffix']:
            logger.warning(f"Item {idx} suffix has no location tokens: {item['suffix']}")
            return False

        if not item['image'] or not isinstance(item['image'], str):
            logger.warning(f"Item {idx} has invalid image field")
            return False

        return True

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def get_item(self, idx: int) -> Tuple[str, str, str]:
        """
        Returns:
            (prefix, suffix, image_id)
            prefix is the raw prefix from JSON, e.g. "<COMMAND> Opens the details page..."
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range (size={len(self.data)})")
        item = self.data[idx]
        return item['prefix'], item['suffix'], item['image']

    def get_format_name(self) -> str:
        return "COMMAND"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def extract_command_text(self, prefix: str) -> str:
        """
        Extract the plain command text from prefix.
        Handles:
          - "<COMMAND> text"  → "text"
          - "text"            → "text"
        """
        if '<COMMAND>' in prefix:
            return prefix.split('<COMMAND>', 1)[1].strip()
        return prefix.strip()

    def extract_bbox(self, suffix: str) -> list:
        """Extract [x1, y1, x2, y2] from suffix location tokens."""
        coords = self.parse_location_tokens(suffix)
        if coords and self.validate_bbox_coordinates(coords):
            return coords
        logger.warning(f"Invalid/missing bbox in suffix: {suffix}")
        return []

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        cmd_lengths = [len(self.extract_command_text(d['prefix']).split()) for d in self.data]
        valid_bboxes = sum(1 for d in self.data if self.extract_bbox(d['suffix']))
        if cmd_lengths:
            stats.update({
                'avg_command_words': sum(cmd_lengths) / len(cmd_lengths),
                'min_command_words': min(cmd_lengths),
                'max_command_words': max(cmd_lengths),
                'valid_bboxes': valid_bboxes,
            })
        return stats


# Keep FUNDatasetProcessor as an alias for backward compatibility
FUNDatasetProcessor = CommandDatasetProcessor
