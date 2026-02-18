"""
DataUtils Package for Florence-2 Training
Supports OD (Object Detection) and COMMAND (Action/Function) formats.
"""

from .base_processor import BaseDatasetProcessor
from .od_processor import ODDatasetProcessor
from .fun_processor import CommandDatasetProcessor, FUNDatasetProcessor  # FUN = alias
from .unified_dataset import UnifiedFlorenceDataset, MixedFormatDataset

__all__ = [
    'BaseDatasetProcessor',
    'ODDatasetProcessor',
    'CommandDatasetProcessor',
    'FUNDatasetProcessor',       # backward compat alias
    'UnifiedFlorenceDataset',
    'MixedFormatDataset',
]
