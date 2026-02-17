"""
Test script for directory-based dataset loading
Tests both OD and FUN formats with individual JSON files
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataUtils import UnifiedFlorenceDataset


def test_fun_directory():
    """Test FUN format loading from directory."""
    print("\n" + "="*60)
    print("Testing FUN Format - Directory Loading")
    print("="*60)
    
    try:
        dataset = UnifiedFlorenceDataset(
            data_path="examples/fun_samples/",  # Directory path
            image_dir="examples/images/",
            format_type="FUN"
        )
        
        print(f"✓ Loaded {len(dataset)} samples from directory")
        
        # Test accessing items
        for i in range(min(len(dataset), 3)):
            prefix, suffix, image_id = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Command: {prefix}")
            print(f"  BBox: {suffix}")
            print(f"  Image: {image_id}")
            
            # Extract command
            command = dataset.processor.extract_command(prefix)
            bbox = dataset.processor.extract_bbox(suffix)
            print(f"  Extracted Command: {command}")
            print(f"  Extracted BBox: {bbox}")
        
        print("\n✓ FUN directory test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_od_directory():
    """Test OD format loading from directory."""
    print("\n" + "="*60)
    print("Testing OD Format - Directory Loading")
    print("="*60)
    
    try:
        dataset = UnifiedFlorenceDataset(
            data_path="examples/od_samples/",  # Directory path
            image_dir="examples/images/",
            format_type="OD"
        )
        
        print(f"✓ Loaded {len(dataset)} samples from directory")
        
        # Test accessing items
        for i in range(min(len(dataset), 3)):
            prefix, suffix, image_id = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Prefix: {prefix}")
            print(f"  Suffix: {suffix}")
            print(f"  Image: {image_id}")
            
            # Extract labels and boxes
            objects = dataset.processor.extract_labels_and_boxes(suffix)
            print(f"  Objects: {len(objects)}")
            for obj in objects:
                print(f"    - {obj['label']}: {obj['bbox']}")
        
        print("\n✓ OD directory test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_detection():
    """Test automatic format detection with directories."""
    print("\n" + "="*60)
    print("Testing Auto-Detection with Directories")
    print("="*60)
    
    try:
        # Test FUN auto-detection
        fun_dataset = UnifiedFlorenceDataset(
            data_path="examples/fun_samples/",
            image_dir="examples/images/",
            format_type="AUTO"
        )
        print(f"FUN directory detected as: {fun_dataset.get_format_name()}")
        
        # Test OD auto-detection
        od_dataset = UnifiedFlorenceDataset(
            data_path="examples/od_samples/",
            image_dir="examples/images/",
            format_type="AUTO"
        )
        print(f"OD directory detected as: {od_dataset.get_format_name()}")
        
        print("\n✓ Auto-detection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all directory-based tests."""
    print("\n" + "="*60)
    print("Directory-Based Dataset Loading Tests")
    print("="*60)
    
    results = {
        "FUN Directory": test_fun_directory(),
        "OD Directory": test_od_directory(),
        "Auto Detection": test_auto_detection()
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
