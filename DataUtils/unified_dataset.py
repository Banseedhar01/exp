"""
Unified Florence Dataset
Supports OD and COMMAND formats with automatic detection.
Supports directory-based or single-file datasets.
"""

import json
import os
import logging
from typing import Tuple, List
from torch.utils.data import Dataset
from .base_processor import BaseDatasetProcessor
from .od_processor import ODDatasetProcessor
from .fun_processor import CommandDatasetProcessor, FUNDatasetProcessor  # FUN = alias

logger = logging.getLogger(__name__)

# Supported format names
FORMAT_OD = "OD"
FORMAT_COMMAND = "COMMAND"
FORMAT_AUTO = "AUTO"


class UnifiedFlorenceDataset(Dataset):
    """
    Unified dataset wrapper for OD and COMMAND formats.
    Accepts either a JSON file or a directory of JSON files.
    format_type: "OD" | "COMMAND" | "FUN" (alias for COMMAND) | "AUTO"
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        image_size: int = 768,
        resize_images: bool = True,
        format_type: str = "AUTO"
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.resize_images = resize_images
        # Normalise aliases
        ft = format_type.upper()
        if ft == "FUN":
            ft = FORMAT_COMMAND
        self.format_type = ft

        self.processor = self._create_processor()
        logger.info(f"UnifiedFlorenceDataset: {self.processor.get_format_name()} | {len(self.processor)} samples")

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    def _detect_format(self, data_path: str) -> str:
        """
        Detect format by reading the first JSON item.
        Works for both a single JSON file and a directory of JSON files.
        """
        try:
            if os.path.isdir(data_path):
                # Read the first JSON file in the directory
                import glob
                files = sorted(glob.glob(os.path.join(data_path, "*.json")))
                if not files:
                    logger.warning("Empty directory, defaulting to OD")
                    return FORMAT_OD
                with open(files[0], 'r', encoding='utf-8') as f:
                    first_item = json.load(f)
                    if isinstance(first_item, list):
                        first_item = first_item[0]
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                first_item = data[0] if isinstance(data, list) else data

            prefix = first_item.get('prefix', '')

            if prefix.strip().startswith('<OD>'):
                logger.info("Auto-detected: OD format")
                return FORMAT_OD

            if '<COMMAND>' in prefix:
                logger.info("Auto-detected: COMMAND format")
                return FORMAT_COMMAND

            # Fallback: if suffix has loc tokens and no <OD>, treat as COMMAND
            if '<loc_' in first_item.get('suffix', '') and not prefix.strip().startswith('<OD>'):
                logger.info("Auto-detected: COMMAND format (by suffix)")
                return FORMAT_COMMAND

            logger.warning("Could not auto-detect format, defaulting to OD")
            return FORMAT_OD

        except Exception as e:
            logger.error(f"Format detection error: {e}, defaulting to OD")
            return FORMAT_OD

    # ------------------------------------------------------------------
    # Processor factory
    # ------------------------------------------------------------------

    def _create_processor(self) -> BaseDatasetProcessor:
        if self.format_type == FORMAT_AUTO:
            self.format_type = self._detect_format(self.data_path)

        if self.format_type == FORMAT_OD:
            return ODDatasetProcessor(self.data_path, self.image_dir, self.image_size, self.resize_images)
        elif self.format_type == FORMAT_COMMAND:
            return CommandDatasetProcessor(self.data_path, self.image_dir, self.image_size, self.resize_images)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}. Use 'OD', 'COMMAND', or 'AUTO'")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.processor)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        """Returns (prefix, suffix, image_id)."""
        return self.processor.get_item(idx)

    def load_image(self, image_id: str):
        return self.processor.load_image(image_id)

    def get_format_name(self) -> str:
        return self.processor.get_format_name()

    def get_statistics(self):
        return self.processor.get_statistics()

    def _format_task_input(self, prefix: str) -> str:
        """Pass prefix as-is to Florence-2 processor."""
        return str(prefix)


class MixedFormatDataset(Dataset):
    """
    Combines multiple UnifiedFlorenceDataset instances for joint training.
    E.g. train on OD + COMMAND data simultaneously.
    """

    def __init__(self, datasets: List[UnifiedFlorenceDataset]):
        self.datasets = datasets
        self._cum_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self._cum_sizes.append(total)

        logger.info(f"MixedFormatDataset: {len(datasets)} datasets, {total} total samples")
        for i, ds in enumerate(datasets):
            logger.info(f"  [{i}] {ds.get_format_name()}: {len(ds)} samples")

    def __len__(self) -> int:
        return self._cum_sizes[-1] if self._cum_sizes else 0

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        ds_idx = next(i for i, s in enumerate(self._cum_sizes) if idx < s)
        local_idx = idx if ds_idx == 0 else idx - self._cum_sizes[ds_idx - 1]
        return self.datasets[ds_idx][local_idx]

    def load_image(self, image_id: str, ds_idx: int = 0):
        return self.datasets[ds_idx].load_image(image_id)

    def get_dataset_for_index(self, idx: int) -> UnifiedFlorenceDataset:
        """Return the sub-dataset that owns this global index."""
        ds_idx = next(i for i, s in enumerate(self._cum_sizes) if idx < s)
        return self.datasets[ds_idx]
