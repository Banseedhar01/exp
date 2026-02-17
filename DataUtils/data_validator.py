"""
Data Validator for Florence-2 Datasets
Provides validation utilities for OD and FUN format datasets
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for Florence-2 dataset formats."""
    
    @staticmethod
    def validate_od_format(data: Dict) -> Tuple[bool, str]:
        """
        Validate OD format data structure.
        
        Args:
            data: Dictionary with OD format data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['prefix', 'suffix', 'image']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate prefix
        if not isinstance(data['prefix'], str):
            return False, "prefix must be a string"
        
        if '<OD>' not in data['prefix']:
            return False, "prefix must contain <OD> token"
        
        # Validate suffix
        if not isinstance(data['suffix'], str):
            return False, "suffix must be a string"
        
        if '<loc_' not in data['suffix']:
            return False, "suffix must contain location tokens (<loc_>)"
        
        # Validate image
        if not isinstance(data['image'], str) or not data['image']:
            return False, "image must be a non-empty string"
        
        return True, ""
    
    @staticmethod
    def validate_fun_format(data: Dict) -> Tuple[bool, str]:
        """
        Validate FUN format data structure.
        
        Args:
            data: Dictionary with FUN format data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['image', 'prefix', 'suffix']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate image
        if not isinstance(data['image'], str) or not data['image']:
            return False, "image must be a non-empty string"
        
        # Validate prefix
        if not isinstance(data['prefix'], str):
            return False, "prefix must be a string"
        
        if '<FUN>' not in data['prefix']:
            return False, "prefix must contain <FUN> token"
        
        if '<CMD>' not in data['prefix']:
            return False, "prefix must contain <CMD> token"
        
        # Validate suffix
        if not isinstance(data['suffix'], str):
            return False, "suffix must be a string"
        
        if '<loc_' not in data['suffix']:
            return False, "suffix must contain location tokens (<loc_>)"
        
        return True, ""
    
    @staticmethod
    def validate_bbox_coordinates(coords: List[int]) -> Tuple[bool, str]:
        """
        Validate bounding box coordinates.
        
        Args:
            coords: List of [x1, y1, x2, y2]
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(coords) != 4:
            return False, f"Expected 4 coordinates, got {len(coords)}"
        
        # Check range (Florence-2 uses 0-999)
        for i, coord in enumerate(coords):
            if not isinstance(coord, int):
                return False, f"Coordinate {i} must be an integer, got {type(coord)}"
            
            if coord < 0 or coord > 999:
                return False, f"Coordinate {i} out of range [0, 999]: {coord}"
        
        # Check x1 < x2 and y1 < y2
        if coords[0] >= coords[2]:
            return False, f"x1 ({coords[0]}) must be < x2 ({coords[2]})"
        
        if coords[1] >= coords[3]:
            return False, f"y1 ({coords[1]}) must be < y2 ({coords[3]})"
        
        return True, ""
    
    @staticmethod
    def validate_json_file(file_path: str, format_type: str = "AUTO") -> Dict[str, Any]:
        """
        Validate entire JSON dataset file.
        
        Args:
            file_path: Path to JSON file
            format_type: Expected format - "OD", "FUN", or "AUTO"
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'total_items': 0,
            'valid_items': 0,
            'invalid_items': 0,
            'errors': [],
            'format': format_type
        }
        
        if not os.path.exists(file_path):
            results['valid'] = False
            results['errors'].append(f"File not found: {file_path}")
            return results
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and single object
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                results['valid'] = False
                results['errors'].append("JSON must be a list or single object")
                return results
            
            results['total_items'] = len(data)
            
            # Auto-detect format from first item if needed
            if format_type == "AUTO" and len(data) > 0:
                first_item = data[0]
                if 'prefix' in first_item and first_item['prefix'].strip().startswith('<OD>'):
                    format_type = "OD"
                elif 'prefix' in first_item and '<FUN>' in first_item['prefix']:
                    format_type = "FUN"
                else:
                    format_type = "OD"  # Default
                
                results['format'] = format_type
                logger.info(f"Auto-detected format: {format_type}")
            
            # Validate each item
            for idx, item in enumerate(data):
                if format_type == "OD":
                    is_valid, error = DataValidator.validate_od_format(item)
                elif format_type == "FUN":
                    is_valid, error = DataValidator.validate_fun_format(item)
                else:
                    is_valid = False
                    error = f"Unknown format: {format_type}"
                
                if is_valid:
                    results['valid_items'] += 1
                else:
                    results['invalid_items'] += 1
                    results['errors'].append(f"Item {idx}: {error}")
            
            # Overall validity
            if results['invalid_items'] > 0:
                results['valid'] = False
            
        except json.JSONDecodeError as e:
            results['valid'] = False
            results['errors'].append(f"JSON parse error: {e}")
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Validation error: {e}")
        
        return results
    
    @staticmethod
    def print_validation_report(results: Dict[str, Any]):
        """
        Print a formatted validation report.
        
        Args:
            results: Validation results dictionary
        """
        print("=" * 60)
        print("Dataset Validation Report")
        print("=" * 60)
        print(f"Format: {results['format']}")
        print(f"Total Items: {results['total_items']}")
        print(f"Valid Items: {results['valid_items']}")
        print(f"Invalid Items: {results['invalid_items']}")
        print(f"Overall Valid: {'✓ YES' if results['valid'] else '✗ NO'}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for i, error in enumerate(results['errors'][:10], 1):  # Show first 10
                print(f"  {i}. {error}")
            
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more errors")
        
        print("=" * 60)


def validate_dataset_cli():
    """Command-line interface for dataset validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Florence-2 dataset files")
    parser.add_argument("file_path", help="Path to JSON dataset file")
    parser.add_argument("--format", choices=["OD", "FUN", "AUTO"], default="AUTO",
                       help="Dataset format (default: AUTO)")
    
    args = parser.parse_args()
    
    results = DataValidator.validate_json_file(args.file_path, args.format)
    DataValidator.print_validation_report(results)
    
    return 0 if results['valid'] else 1


if __name__ == "__main__":
    import sys
    sys.exit(validate_dataset_cli())
