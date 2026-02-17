"""
Object Detection (OD) Dataset Processor for Florence-2
Handles dataset format: {"prefix": "<OD>", "suffix": "label<loc_x1><loc_y1><loc_x2><loc_y2>...", "image": "<image_name>"}
Supports both single JSON file and directory of individual JSON files
"""

import json
import logging
import os
import glob
from typing import Tuple, List, Dict, Any
from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class ODDatasetProcessor(BaseDatasetProcessor):
    """
    Processor for Object Detection format datasets.
    
    Expected JSON format:
    {
        "prefix": "<OD>",
        "suffix": "label<loc_x1><loc_y1><loc_x2><loc_y2>...",
        "image": "image_name.png"
    }
    """
    
    def __init__(self, data_path: str, image_dir: str, image_size: int = 768, resize_images: bool = True):
        """
        Initialize OD dataset processor.
        
        Args:
            data_path: Path to JSON file OR directory containing JSON files
            image_dir: Directory containing images
            image_size: Target size for image resizing
            resize_images: Whether to resize images
        """
        super().__init__(data_path, image_dir, image_size, resize_images)
    
    def _load_data(self):
        """Load and validate OD format JSON data from file or directory."""
        logger.info(f"Loading OD format data from {self.data_path}")
        
        try:
            # Check if data_path is a directory
            if os.path.isdir(self.data_path):
                logger.info(f"Loading from directory: {self.data_path}")
                self._load_from_directory()
            else:
                logger.info(f"Loading from single file: {self.data_path}")
                self._load_from_file()
            
            logger.info(f"Loaded {len(self.data)} valid OD samples")
            
        except Exception as e:
            logger.error(f"Error loading OD data: {e}")
            raise
    
    def _load_from_file(self):
        """Load data from a single JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Handle both list of objects and single object
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            
            # Validate and filter data
            for idx, item in enumerate(raw_data):
                if self._validate_od_item(item, idx):
                    self.data.append(item)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {self.data_path}: {e}")
            raise
    
    def _load_from_directory(self):
        """Load data from directory of individual JSON files."""
        json_files = glob.glob(os.path.join(self.data_path, "*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.data_path}")
            return
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                
                if self._validate_od_item(item, len(self.data)):
                    self.data.append(item)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing {json_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue
    
    def _validate_od_item(self, item: Dict, idx: int) -> bool:
        """
        Validate a single OD format item.
        
        Args:
            item: Dictionary containing the data item
            idx: Index for logging purposes
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['prefix', 'suffix', 'image']
        
        # Check required fields
        for field in required_fields:
            if field not in item:
                logger.warning(f"Item {idx} missing required field: {field}")
                return False
        
        # Validate prefix contains <OD>
        if '<OD>' not in item['prefix']:
            logger.warning(f"Item {idx} prefix does not contain <OD>: {item['prefix']}")
            return False
        
        # Validate suffix contains location tokens
        if '<loc_' not in item['suffix']:
            logger.warning(f"Item {idx} suffix does not contain location tokens: {item['suffix']}")
            return False
        
        # Validate image field is not empty
        if not item['image'] or not isinstance(item['image'], str):
            logger.warning(f"Item {idx} has invalid image field: {item['image']}")
            return False
        
        return True
    
    def get_item(self, idx: int) -> Tuple[str, str, str]:
        """
        Get a single OD format item.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (prefix, suffix, image_id)
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        item = self.data[idx]
        
        prefix = item['prefix']
        suffix = item['suffix']
        image_id = item['image']
        
        return prefix, suffix, image_id
    
    def get_format_name(self) -> str:
        """Return the format name."""
        return "OD"
    
    def extract_labels_and_boxes(self, suffix: str) -> List[Dict[str, Any]]:
        """
        Extract labels and bounding boxes from suffix.
        
        Args:
            suffix: Suffix string like "label1<loc_x1><loc_y1><loc_x2><loc_y2>label2<loc_x1>..."
            
        Returns:
            List of dictionaries with 'label' and 'bbox' keys
        """
        import re
        
        # Pattern to match label followed by 4 location tokens
        pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        matches = re.findall(pattern, suffix)
        
        results = []
        for match in matches:
            label = match[0].strip()
            bbox = [int(match[1]), int(match[2]), int(match[3]), int(match[4])]
            
            if self.validate_bbox_coordinates(bbox):
                results.append({
                    'label': label,
                    'bbox': bbox
                })
            else:
                logger.warning(f"Invalid bbox coordinates for label '{label}': {bbox}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get OD dataset statistics."""
        stats = super().get_statistics()
        
        # Count total objects across all samples
        total_objects = 0
        unique_labels = set()
        
        for item in self.data:
            objects = self.extract_labels_and_boxes(item['suffix'])
            total_objects += len(objects)
            for obj in objects:
                unique_labels.add(obj['label'])
        
        stats.update({
            'total_objects': total_objects,
            'unique_labels': len(unique_labels),
            'avg_objects_per_image': total_objects / len(self.data) if self.data else 0
        })
        
        return stats
