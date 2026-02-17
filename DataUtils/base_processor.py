"""
Base Dataset Processor for Florence-2 Training
Provides abstract interface and common utilities for all dataset processors
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class BaseDatasetProcessor(ABC):
    """
    Abstract base class for dataset processors.
    All format-specific processors should inherit from this class.
    """
    
    def __init__(self, data_path: str, image_dir: str, image_size: int = 768, resize_images: bool = True):
        """
        Initialize the dataset processor.
        
        Args:
            data_path: Path to the dataset file (JSON or CSV) OR directory containing JSON files
            image_dir: Directory containing images
            image_size: Target size for image resizing
            resize_images: Whether to resize images
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.resize_images = resize_images
        self.data = []
        
        # Validate paths
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Load data
        self._load_data()
        logger.info(f"{self.__class__.__name__} initialized with {len(self.data)} samples")
    
    @abstractmethod
    def _load_data(self):
        """Load and parse the dataset file. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_item(self, idx: int) -> Tuple[str, str, str]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (prefix, suffix, image_id)
        """
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Return the name of the dataset format (e.g., 'OD', 'FUN')"""
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def load_image(self, image_id: str) -> Image.Image:
        """
        Load and optionally resize an image.
        
        Args:
            image_id: Image filename or ID
            
        Returns:
            PIL Image object
        """
        image_path = os.path.join(self.image_dir, image_id)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.resize_images:
                image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def validate_bbox_coordinates(self, coords: List[int]) -> bool:
        """
        Validate bounding box coordinates.
        
        Args:
            coords: List of coordinates [x1, y1, x2, y2]
            
        Returns:
            True if valid, False otherwise
        """
        if len(coords) != 4:
            return False
        
        # Check if all coordinates are in valid range (0-999 for Florence-2)
        if not all(0 <= c <= 999 for c in coords):
            return False
        
        # Check if x1 < x2 and y1 < y2
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            return False
        
        return True
    
    def format_location_tokens(self, coords: List[int]) -> str:
        """
        Format coordinates as location tokens.
        
        Args:
            coords: List of coordinates [x1, y1, x2, y2]
            
        Returns:
            Formatted string like "<loc_x1><loc_y1><loc_x2><loc_y2>"
        """
        if not self.validate_bbox_coordinates(coords):
            logger.warning(f"Invalid coordinates: {coords}")
        
        return f"<loc_{coords[0]}><loc_{coords[1]}><loc_{coords[2]}><loc_{coords[3]}>"
    
    def parse_location_tokens(self, suffix: str) -> List[int]:
        """
        Parse location tokens from suffix string.
        
        Args:
            suffix: String containing location tokens like "<loc_756><loc_467><loc_981><loc_507>"
            
        Returns:
            List of coordinates [x1, y1, x2, y2]
        """
        import re
        pattern = r'<loc_(\d+)>'
        matches = re.findall(pattern, suffix)
        
        if len(matches) < 4:
            logger.warning(f"Could not parse 4 coordinates from: {suffix}")
            return []
        
        coords = [int(m) for m in matches[:4]]
        return coords
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'format': self.get_format_name(),
            'total_samples': len(self.data),
            'data_path': self.data_path,
            'image_dir': self.image_dir,
            'image_size': self.image_size,
            'resize_images': self.resize_images
        }
