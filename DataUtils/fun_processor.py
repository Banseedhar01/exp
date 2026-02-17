"""
Function/Action (FUN) Dataset Processor for Florence-2
Handles dataset format: {"image": "20.png", "prefix": "command text", "suffix": "<loc_756><loc_467><loc_981><loc_507>"}
Supports both single JSON file and directory of individual JSON files
"""

import json
import logging
import os
import glob
from typing import Tuple, Dict, Any
from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class FUNDatasetProcessor(BaseDatasetProcessor):
    """
    Processor for Function/Action format datasets.
    
    Expected JSON format (simplified):
    {
        "image": "image_name.png",
        "prefix": "Command text here",
        "suffix": "<loc_x1><loc_y1><loc_x2><loc_y2>"
    }
    
    Also supports legacy format with <FUN> and <CMD> tokens.
    """
    
    def __init__(self, data_path: str, image_dir: str, image_size: int = 768, resize_images: bool = True):
        """
        Initialize FUN dataset processor.
        
        Args:
            data_path: Path to JSON file OR directory containing JSON files
            image_dir: Directory containing images
            image_size: Target size for image resizing
            resize_images: Whether to resize images
        """
        super().__init__(data_path, image_dir, image_size, resize_images)
    
    def _load_data(self):
        """Load and validate FUN format JSON data from file or directory."""
        logger.info(f"Loading FUN format data from {self.data_path}")
        
        try:
            # Check if data_path is a directory
            if os.path.isdir(self.data_path):
                logger.info(f"Loading from directory: {self.data_path}")
                self._load_from_directory()
            else:
                logger.info(f"Loading from single file: {self.data_path}")
                self._load_from_file()
            
            logger.info(f"Loaded {len(self.data)} valid FUN samples")
            
        except Exception as e:
            logger.error(f"Error loading FUN data: {e}")
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
                if self._validate_fun_item(item, idx):
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
                
                if self._validate_fun_item(item, len(self.data)):
                    self.data.append(item)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing {json_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue
    
    def _validate_fun_item(self, item: Dict, idx: int) -> bool:
        """
        Validate a single FUN format item.
        Supports both new format (plain text) and legacy format (with FUN/CMD tokens).
        
        Args:
            item: Dictionary containing the data item
            idx: Index for logging purposes
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['image', 'prefix', 'suffix']
        
        # Check required fields
        for field in required_fields:
            if field not in item:
                logger.warning(f"Item {idx} missing required field: {field}")
                return False
        
        # Validate prefix is a non-empty string (command text)
        if not isinstance(item['prefix'], str) or not item['prefix'].strip():
            logger.warning(f"Item {idx} prefix must be non-empty string: {item['prefix']}")
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
        Get a single FUN format item.
        
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
        return "FUN"
    
    def extract_command(self, prefix: str) -> str:
        """
        Extract command text from prefix.
        Supports both formats:
        - "<COMMAND> your text" (with token)
        - "your text" (plain text)
        
        Args:
            prefix: Prefix string - either "<COMMAND> text" or plain "text"
            
        Returns:
            Command text
        """
        # Check if format with <COMMAND> token
        if '<COMMAND>' in prefix:
            parts = prefix.split('<COMMAND>')
            if len(parts) > 1:
                command = parts[1].strip()
                return command
        
        # Check for legacy <CMD> token (backward compatibility)
        if '<CMD>' in prefix:
            parts = prefix.split('<CMD>')
            if len(parts) > 1:
                command = parts[1].strip()
                return command
        
        # Plain text command (no tokens)
        command = prefix.strip()
        return command
    
    def extract_bbox(self, suffix: str) -> list:
        """
        Extract bounding box coordinates from suffix.
        
        Args:
            suffix: Suffix string like "<loc_756><loc_467><loc_981><loc_507>"
            
        Returns:
            List of coordinates [x1, y1, x2, y2]
        """
        coords = self.parse_location_tokens(suffix)
        
        if coords and self.validate_bbox_coordinates(coords):
            return coords
        else:
            logger.warning(f"Invalid or missing bbox in suffix: {suffix}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get FUN dataset statistics."""
        stats = super().get_statistics()
        
        # Analyze command lengths
        command_lengths = []
        valid_bboxes = 0
        
        for item in self.data:
            command = self.extract_command(item['prefix'])
            command_lengths.append(len(command.split()))
            
            bbox = self.extract_bbox(item['suffix'])
            if bbox:
                valid_bboxes += 1
        
        if command_lengths:
            stats.update({
                'avg_command_length': sum(command_lengths) / len(command_lengths),
                'min_command_length': min(command_lengths),
                'max_command_length': max(command_lengths),
                'valid_bboxes': valid_bboxes,
                'bbox_coverage': valid_bboxes / len(self.data) if self.data else 0
            })
        
        return stats
