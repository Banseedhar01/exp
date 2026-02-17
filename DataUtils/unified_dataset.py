"""
Unified Florence Dataset
Provides a unified interface for multiple dataset formats with automatic format detection
"""

import json
import logging
from typing import Tuple, Optional, List
from torch.utils.data import Dataset
from .base_processor import BaseDatasetProcessor
from .od_processor import ODDatasetProcessor
from .fun_processor import FUNDatasetProcessor

logger = logging.getLogger(__name__)


class UnifiedFlorenceDataset(Dataset):
    """
    Unified dataset wrapper that supports multiple formats.
    Automatically detects format or uses specified format.
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        image_size: int = 768,
        resize_images: bool = True,
        format_type: str = "AUTO"
    ):
        """
        Initialize unified dataset.
        
        Args:
            data_path: Path to dataset file (JSON)
            image_dir: Directory containing images
            image_size: Target size for image resizing
            resize_images: Whether to resize images
            format_type: Dataset format - "OD", "FUN", or "AUTO" for automatic detection
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.resize_images = resize_images
        self.format_type = format_type.upper()
        
        # Initialize the appropriate processor
        self.processor = self._create_processor()
        
        logger.info(f"UnifiedFlorenceDataset initialized with {self.processor.get_format_name()} format")
        logger.info(f"Total samples: {len(self.processor)}")
    
    def _detect_format(self, data_path: str) -> str:
        """
        Automatically detect dataset format by examining the first item.
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            Format type: "OD" or "FUN"
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get first item
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
            elif isinstance(data, dict):
                first_item = data
            else:
                raise ValueError("Invalid JSON structure")
            
            # Check for format indicators
            if 'prefix' in first_item:
                prefix = first_item['prefix']
                
                # OD format: prefix starts with <OD>
                if prefix.strip().startswith('<OD>'):
                    logger.info("Auto-detected format: OD (Object Detection)")
                    return "OD"
                
                # FUN format: prefix contains <FUN> and <CMD>
                elif '<FUN>' in prefix and '<CMD>' in prefix:
                    logger.info("Auto-detected format: FUN (Function/Action)")
                    return "FUN"
            
            # Default to OD if unclear
            logger.warning("Could not auto-detect format, defaulting to OD")
            return "OD"
            
        except Exception as e:
            logger.error(f"Error detecting format: {e}")
            logger.warning("Defaulting to OD format")
            return "OD"
    
    def _create_processor(self) -> BaseDatasetProcessor:
        """
        Create the appropriate dataset processor based on format type.
        
        Returns:
            Dataset processor instance
        """
        # Auto-detect format if needed
        if self.format_type == "AUTO":
            detected_format = self._detect_format(self.data_path)
            self.format_type = detected_format
        
        # Create processor based on format
        if self.format_type == "OD":
            return ODDatasetProcessor(
                self.data_path,
                self.image_dir,
                self.image_size,
                self.resize_images
            )
        elif self.format_type == "FUN":
            return FUNDatasetProcessor(
                self.data_path,
                self.image_dir,
                self.image_size,
                self.resize_images
            )
        else:
            raise ValueError(f"Unsupported format type: {self.format_type}. Use 'OD', 'FUN', or 'AUTO'")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.processor)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (prefix, suffix, image_id)
        """
        return self.processor.get_item(idx)
    
    def load_image(self, image_id: str):
        """
        Load an image.
        
        Args:
            image_id: Image filename
            
        Returns:
            PIL Image object
        """
        return self.processor.load_image(image_id)
    
    def get_format_name(self) -> str:
        """Return the dataset format name."""
        return self.processor.get_format_name()
    
    def get_statistics(self):
        """Get dataset statistics."""
        return self.processor.get_statistics()
    
    def _format_task_input(self, prefix: str) -> str:
        """
        Format the task input for Florence-2 processor.
        
        Args:
            prefix: Raw prefix from dataset
            
        Returns:
            Formatted task input string
        """
        # For Florence-2, the prefix is already in the correct format
        # Just ensure it's a string
        return str(prefix)


class MixedFormatDataset(Dataset):
    """
    Dataset that combines multiple formats.
    Useful for training on both OD and FUN data simultaneously.
    """
    
    def __init__(self, datasets: List[UnifiedFlorenceDataset]):
        """
        Initialize mixed format dataset.
        
        Args:
            datasets: List of UnifiedFlorenceDataset instances
        """
        self.datasets = datasets
        self.cumulative_sizes = self._calculate_cumulative_sizes()
        
        logger.info(f"MixedFormatDataset initialized with {len(datasets)} datasets")
        logger.info(f"Total samples: {self.cumulative_sizes[-1] if self.cumulative_sizes else 0}")
        
        # Log format distribution
        for i, dataset in enumerate(datasets):
            logger.info(f"  Dataset {i}: {dataset.get_format_name()} format, {len(dataset)} samples")
    
    def _calculate_cumulative_sizes(self) -> List[int]:
        """Calculate cumulative sizes for indexing."""
        cumulative = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            cumulative.append(total)
        return cumulative
    
    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        """
        Get item from the appropriate dataset.
        
        Args:
            idx: Global index
            
        Returns:
            Tuple of (prefix, suffix, image_id)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break
        
        # Calculate local index within the dataset
        local_idx = idx
        if dataset_idx > 0:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][local_idx]
    
    def load_image(self, image_id: str, dataset_idx: int = 0):
        """
        Load image from specified dataset.
        
        Args:
            image_id: Image filename
            dataset_idx: Index of dataset to load from
            
        Returns:
            PIL Image object
        """
        return self.datasets[dataset_idx].load_image(image_id)
