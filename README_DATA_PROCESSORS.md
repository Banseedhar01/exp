# Florence-2 Modular Data Processor

A modular, extensible data processing system for Florence-2 training that supports multiple dataset formats.

## Supported Formats

### 1. OD Format (Object Detection)
```json
{
  "prefix": "<OD>",
  "suffix": "label<loc_x1><loc_y1><loc_x2><loc_y2>...",
  "image": "image_name.png"
}
```

### 2. FUN Format (Function/Action)
```json
{
  "image": "20.png",
  "prefix": "<FUN>\n<CMD> Subscribe to the podcast",
  "suffix": "<loc_756><loc_467><loc_981><loc_507>"
}
```

## Quick Start

### 1. Prepare Your Dataset

Create a JSON file with your data in either OD or FUN format:

**OD Format Example (`od_dataset.json`):**
```json
[
  {
    "prefix": "<OD>",
    "suffix": "button<loc_100><loc_200><loc_300><loc_400>",
    "image": "screenshot1.png"
  },
  {
    "prefix": "<OD>",
    "suffix": "text<loc_50><loc_100><loc_250><loc_150>icon<loc_300><loc_100><loc_350><loc_150>",
    "image": "screenshot2.png"
  }
]
```

**FUN Format Example (`fun_dataset.json`):**
```json
[
  {
    "image": "20.png",
    "prefix": "<FUN>\n<CMD> Subscribe to the podcast",
    "suffix": "<loc_756><loc_467><loc_981><loc_507>"
  },
  {
    "image": "21.png",
    "prefix": "<FUN>\n<CMD> Click the login button",
    "suffix": "<loc_100><loc_200><loc_300><loc_250>"
  }
]
```

### 2. Validate Your Dataset

```bash
python DataUtils/data_validator.py your_dataset.json --format AUTO
```

### 3. Configure Training

Edit `Config.py`:

```python
class Config:
    # Set your dataset path
    DATASET_JSON_PATH = "data/your_dataset.json"
    
    # Set format: "OD", "FUN", or "AUTO" for automatic detection
    DATASET_FORMAT = "AUTO"
    
    # Set image directory
    image_dir = "path/to/your/images/"
    
    # Other settings...
    BATCH_SIZE = 2
    EPOCHS = 7
    LEARNING_RATE = 1e-5
```

### 4. Run Training

```bash
# Single GPU
python train_florence_new.py

# Multi-GPU with Accelerate
accelerate launch train_florence_new.py
```

## Architecture

```
DataUtils/
├── __init__.py                 # Package exports
├── base_processor.py           # Abstract base class
├── od_processor.py             # OD format processor
├── fun_processor.py            # FUN format processor
├── unified_dataset.py          # Unified dataset wrapper
└── data_validator.py           # Validation utilities
```

### Key Components

#### BaseDatasetProcessor
Abstract base class providing:
- Image loading and resizing
- Bounding box validation
- Location token formatting/parsing
- Common utilities

#### ODDatasetProcessor
Handles Object Detection format:
- JSON parsing and validation
- Label and bbox extraction
- Multiple objects per image support

#### FUNDatasetProcessor
Handles Function/Action format:
- Command text extraction
- Single bbox per image
- FUN/CMD token handling

#### UnifiedFlorenceDataset
Unified PyTorch Dataset:
- Automatic format detection
- Single interface for all formats
- Easy integration with DataLoader

## Usage Examples

### Basic Usage

```python
from DataUtils import UnifiedFlorenceDataset

# Create dataset (auto-detects format)
dataset = UnifiedFlorenceDataset(
    data_path="data/my_dataset.json",
    image_dir="data/images/",
    format_type="AUTO"
)

# Get item
prefix, suffix, image_id = dataset[0]
image = dataset.load_image(image_id)

# Get statistics
stats = dataset.get_statistics()
print(f"Format: {stats['format']}")
print(f"Total samples: {stats['total_samples']}")
```

### Using Specific Format

```python
# Force OD format
od_dataset = UnifiedFlorenceDataset(
    data_path="data/od_data.json",
    image_dir="data/images/",
    format_type="OD"
)

# Force FUN format
fun_dataset = UnifiedFlorenceDataset(
    data_path="data/fun_data.json",
    image_dir="data/images/",
    format_type="FUN"
)
```

### Mixed Format Training

```python
from DataUtils import UnifiedFlorenceDataset, MixedFormatDataset

# Create datasets
od_dataset = UnifiedFlorenceDataset(
    data_path="data/od_data.json",
    image_dir="data/images/",
    format_type="OD"
)

fun_dataset = UnifiedFlorenceDataset(
    data_path="data/fun_data.json",
    image_dir="data/images/",
    format_type="FUN"
)

# Combine them
mixed_dataset = MixedFormatDataset([od_dataset, fun_dataset])

# Use in training
train_loader = DataLoader(mixed_dataset, batch_size=2)
```

### Validation

```python
from DataUtils.data_validator import DataValidator

# Validate dataset file
results = DataValidator.validate_json_file(
    "data/my_dataset.json",
    format_type="AUTO"
)

# Print report
DataValidator.print_validation_report(results)

# Check if valid
if results['valid']:
    print("Dataset is valid!")
else:
    print(f"Found {results['invalid_items']} invalid items")
    for error in results['errors']:
        print(f"  - {error}")
```

## Coordinate System

Florence-2 uses normalized coordinates in the range [0, 999]:
- `<loc_0>` = 0% of image dimension
- `<loc_999>` = 100% of image dimension

Example: For a 1000x1000 image, `<loc_500>` = pixel 500

## Adding New Formats

To add a new format:

1. Create a new processor class inheriting from `BaseDatasetProcessor`
2. Implement required methods: `_load_data()`, `get_item()`, `get_format_name()`
3. Add format detection logic to `UnifiedFlorenceDataset._detect_format()`
4. Update `Config.py` with new format option

Example:

```python
from DataUtils.base_processor import BaseDatasetProcessor

class MyCustomProcessor(BaseDatasetProcessor):
    def _load_data(self):
        # Load your custom format
        pass
    
    def get_item(self, idx):
        # Return (prefix, suffix, image_id)
        pass
    
    def get_format_name(self):
        return "CUSTOM"
```

## Backward Compatibility

The new system is fully backward compatible with existing code:

```python
# Old way (still works)
from DataUtils.ActionDataset import ActionDataset, FlorenceActionDataset
action_dataset = ActionDataset(Config.commands_path)
florence_dataset = FlorenceActionDataset(action_dataset)

# New way (recommended)
from DataUtils import UnifiedFlorenceDataset
dataset = UnifiedFlorenceDataset(
    data_path=Config.DATASET_JSON_PATH,
    image_dir=Config.image_dir,
    format_type="AUTO"
)
```

## Troubleshooting

### Dataset Validation Fails

Run the validator to see specific errors:
```bash
python DataUtils/data_validator.py your_dataset.json
```

Common issues:
- Missing required fields (`prefix`, `suffix`, `image`)
- Invalid coordinate ranges (must be 0-999)
- Invalid bbox (x1 >= x2 or y1 >= y2)
- Missing location tokens in suffix

### Format Not Detected

Ensure your JSON has the correct structure:
- OD: `prefix` starts with `<OD>`
- FUN: `prefix` contains both `<FUN>` and `<CMD>`

### Image Not Found

Check:
- Image directory path in `Config.py`
- Image filenames in JSON match actual files
- File extensions are correct

## License

Same as Florence-2 base model.
