# Directory-Based Dataset Loading Guide

## Overview

The Florence-2 data processors now support **directory-based loading** where each sample is stored as a separate JSON file. This is ideal for large datasets where individual files are easier to manage.

## Dataset Formats

### FUN Format (Simplified - No Tokens Required)

**New Format** (Plain text command):
```json
{
  "image": "screenshot1.png",
  "prefix": "Click the subscribe button",
  "suffix": "<loc_756><loc_467><loc_981><loc_507>"
}
```

**Legacy Format** (Still supported):
```json
{
  "image": "screenshot1.png",
  "prefix": "<FUN>\n<CMD> Click the subscribe button",
  "suffix": "<loc_756><loc_467><loc_981><loc_507>"
}
```

### OD Format

```json
{
  "prefix": "<OD>",
  "suffix": "button<loc_100><loc_200><loc_300><loc_250>",
  "image": "screenshot1.png"
}
```

## Directory Structure

### Option 1: Directory of Individual JSON Files

```
your_dataset/
├── fun_data/
│   ├── sample001.json
│   ├── sample002.json
│   ├── sample003.json
│   └── ...
├── od_data/
│   ├── sample001.json
│   ├── sample002.json
│   └── ...
└── images/
    ├── screenshot1.png
    ├── screenshot2.png
    └── ...
```

### Option 2: Single JSON File (Still Supported)

```
your_dataset/
├── fun_data.json       # Array of all samples
├── od_data.json        # Array of all samples
└── images/
    ├── screenshot1.png
    └── ...
```

## Usage Examples

### Load from Directory

```python
from DataUtils import UnifiedFlorenceDataset

# FUN format from directory
fun_dataset = UnifiedFlorenceDataset(
    data_path="path/to/fun_data/",      # Directory path
    image_dir="path/to/images/",
    format_type="FUN"
)

# OD format from directory
od_dataset = UnifiedFlorenceDataset(
    data_path="path/to/od_data/",       # Directory path
    image_dir="path/to/images/",
    format_type="OD"
)
```

### Load from Single File

```python
# FUN format from file
fun_dataset = UnifiedFlorenceDataset(
    data_path="path/to/fun_data.json",  # File path
    image_dir="path/to/images/",
    format_type="FUN"
)
```

### Auto-Detection

```python
# Automatically detects format
dataset = UnifiedFlorenceDataset(
    data_path="path/to/data/",          # Can be file or directory
    image_dir="path/to/images/",
    format_type="AUTO"
)

print(f"Detected format: {dataset.get_format_name()}")
```

## Configuration for Training

Update `Config.py`:

```python
class Config:
    # For directory-based loading
    DATASET_JSON_PATH = "path/to/fun_data/"  # Directory path
    DATASET_FORMAT = "FUN"  # or "OD" or "AUTO"
    image_dir = "path/to/images/"
    
    # Other settings...
    BATCH_SIZE = 2
    EPOCHS = 7
```

## Creating Your Dataset

### FUN Format (Recommended - Simplified)

For each sample, create a JSON file:

**sample001.json:**
```json
{
  "image": "screenshot001.png",
  "prefix": "Click the login button",
  "suffix": "<loc_100><loc_200><loc_300><loc_250>"
}
```

**sample002.json:**
```json
{
  "image": "screenshot002.png",
  "prefix": "Open settings menu",
  "suffix": "<loc_50><loc_50><loc_150><loc_100>"
}
```

### OD Format

**sample001.json:**
```json
{
  "prefix": "<OD>",
  "suffix": "button<loc_100><loc_200><loc_300><loc_250>text<loc_50><loc_100><loc_250><loc_150>",
  "image": "screenshot001.png"
}
```

## Key Changes

### ✅ What's New

1. **Directory-based loading**: Point to a directory instead of a single file
2. **Simplified FUN format**: No need for `<FUN>` or `<CMD>` tokens - just plain text commands
3. **Backward compatible**: Legacy formats still work
4. **Auto-detection**: Works with both files and directories

### 📝 Migration Guide

**Old FUN format:**
```json
{
  "image": "img.png",
  "prefix": "<FUN>\n<CMD> Click button",
  "suffix": "<loc_100><loc_200><loc_300><loc_250>"
}
```

**New FUN format (recommended):**
```json
{
  "image": "img.png",
  "prefix": "Click button",
  "suffix": "<loc_100><loc_200><loc_300><loc_250>"
}
```

Both formats work! The processor automatically handles both.

## Testing

Test directory loading:

```bash
cd d:\Desktop\Samsung\Y26\train_florence
python test_directory_loading.py
```

## Example Directory Structure

See `examples/` folder:

```
examples/
├── fun_samples/          # Directory of FUN format JSONs
│   ├── sample1.json
│   └── sample2.json
├── od_samples/           # Directory of OD format JSONs
│   ├── sample1.json
│   └── sample2.json
└── images/               # Shared image directory
    └── (placeholder images)
```

## Benefits of Directory-Based Loading

✅ **Easier to manage**: One file per sample  
✅ **Parallel processing**: Can process files in parallel  
✅ **Incremental updates**: Add/remove samples easily  
✅ **Better for large datasets**: No single huge JSON file  
✅ **Version control friendly**: Git diffs work better with individual files

## Training Example

```python
from Config import Config
from DataUtils import UnifiedFlorenceDataset

# Load from directory
dataset = UnifiedFlorenceDataset(
    data_path=Config.DATASET_JSON_PATH,  # Can be directory or file
    image_dir=Config.image_dir,
    format_type=Config.DATASET_FORMAT
)

print(f"Loaded {len(dataset)} samples")
print(f"Format: {dataset.get_format_name()}")

# Use in training
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)
```
