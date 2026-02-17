"""
Test script for the modular data processors
Demonstrates loading and using both OD and FUN format datasets
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataUtils import UnifiedFlorenceDataset, MixedFormatDataset
from DataUtils.data_validator import DataValidator


def test_od_format():
    """Test OD format dataset loading."""
    print("\n" + "="*60)
    print("Testing OD Format Dataset")
    print("="*60)
    
    # Validate first
    print("\n1. Validating dataset...")
    results = DataValidator.validate_json_file(
        "examples/sample_od_dataset.json",
        format_type="OD"
    )
    DataValidator.print_validation_report(results)
    
    if not results['valid']:
        print("❌ Validation failed!")
        return False
    
    # Load dataset
    print("\n2. Loading dataset...")
    try:
        dataset = UnifiedFlorenceDataset(
            data_path="examples/sample_od_dataset.json",
            image_dir="examples/images/",
            format_type="OD"
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Get statistics
    print("\n3. Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test item access
    print("\n4. Testing item access...")
    try:
        prefix, suffix, image_id = dataset[0]
        print(f"   Sample 0:")
        print(f"     Prefix: {prefix}")
        print(f"     Suffix: {suffix}")
        print(f"     Image: {image_id}")
    except Exception as e:
        print(f"❌ Error accessing item: {e}")
        return False
    
    print("\n✓ OD format test passed!")
    return True


def test_fun_format():
    """Test FUN format dataset loading."""
    print("\n" + "="*60)
    print("Testing FUN Format Dataset")
    print("="*60)
    
    # Validate first
    print("\n1. Validating dataset...")
    results = DataValidator.validate_json_file(
        "examples/sample_fun_dataset.json",
        format_type="FUN"
    )
    DataValidator.print_validation_report(results)
    
    if not results['valid']:
        print("❌ Validation failed!")
        return False
    
    # Load dataset
    print("\n2. Loading dataset...")
    try:
        dataset = UnifiedFlorenceDataset(
            data_path="examples/sample_fun_dataset.json",
            image_dir="examples/images/",
            format_type="FUN"
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Get statistics
    print("\n3. Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test item access
    print("\n4. Testing item access...")
    try:
        prefix, suffix, image_id = dataset[0]
        print(f"   Sample 0:")
        print(f"     Prefix: {prefix}")
        print(f"     Suffix: {suffix}")
        print(f"     Image: {image_id}")
        
        # Extract command
        from DataUtils.fun_processor import FUNDatasetProcessor
        command = dataset.processor.extract_command(prefix)
        bbox = dataset.processor.extract_bbox(suffix)
        print(f"     Command: {command}")
        print(f"     BBox: {bbox}")
    except Exception as e:
        print(f"❌ Error accessing item: {e}")
        return False
    
    print("\n✓ FUN format test passed!")
    return True


def test_auto_detection():
    """Test automatic format detection."""
    print("\n" + "="*60)
    print("Testing Automatic Format Detection")
    print("="*60)
    
    # Test OD auto-detection
    print("\n1. Testing OD auto-detection...")
    try:
        dataset = UnifiedFlorenceDataset(
            data_path="examples/sample_od_dataset.json",
            image_dir="examples/images/",
            format_type="AUTO"
        )
        detected_format = dataset.get_format_name()
        print(f"   Detected format: {detected_format}")
        
        if detected_format != "OD":
            print(f"❌ Expected OD, got {detected_format}")
            return False
        print("   ✓ Correct!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test FUN auto-detection
    print("\n2. Testing FUN auto-detection...")
    try:
        dataset = UnifiedFlorenceDataset(
            data_path="examples/sample_fun_dataset.json",
            image_dir="examples/images/",
            format_type="AUTO"
        )
        detected_format = dataset.get_format_name()
        print(f"   Detected format: {detected_format}")
        
        if detected_format != "FUN":
            print(f"❌ Expected FUN, got {detected_format}")
            return False
        print("   ✓ Correct!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n✓ Auto-detection test passed!")
    return True


def test_mixed_dataset():
    """Test mixed format dataset."""
    print("\n" + "="*60)
    print("Testing Mixed Format Dataset")
    print("="*60)
    
    try:
        # Create individual datasets
        od_dataset = UnifiedFlorenceDataset(
            data_path="examples/sample_od_dataset.json",
            image_dir="examples/images/",
            format_type="OD"
        )
        
        fun_dataset = UnifiedFlorenceDataset(
            data_path="examples/sample_fun_dataset.json",
            image_dir="examples/images/",
            format_type="FUN"
        )
        
        # Combine them
        mixed_dataset = MixedFormatDataset([od_dataset, fun_dataset])
        
        print(f"✓ Mixed dataset created: {len(mixed_dataset)} total samples")
        print(f"   OD samples: {len(od_dataset)}")
        print(f"   FUN samples: {len(fun_dataset)}")
        
        # Test accessing items from both datasets
        print("\n   Testing item access...")
        prefix, suffix, image_id = mixed_dataset[0]
        print(f"   Item 0 (should be OD): {image_id}")
        
        prefix, suffix, image_id = mixed_dataset[len(od_dataset)]
        print(f"   Item {len(od_dataset)} (should be FUN): {image_id}")
        
        print("\n✓ Mixed dataset test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Florence-2 Data Processor Test Suite")
    print("="*60)
    
    results = {
        "OD Format": test_od_format(),
        "FUN Format": test_fun_format(),
        "Auto Detection": test_auto_detection(),
        "Mixed Dataset": test_mixed_dataset()
    }
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
