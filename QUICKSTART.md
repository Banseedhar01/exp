# Florence-2 Data Processors - Updated

## Quick Start

### Your Dataset Structure

```
your_project/
├── fun_commands/          # Directory with individual JSON files
│   ├── 001.json
│   ├── 002.json
│   └── ...
├── images/                # All your images
│   ├── screenshot1.png
│   ├── screenshot2.png
│   └── ...
```

### JSON Format (Simplified)

**Each JSON file** (e.g., `001.json`):
```json
{
  "image": "screenshot1.png",
  "prefix": "Click the subscribe button",
  "suffix": "<loc_756><loc_467><loc_981><loc_507>"
}
```

That's it! No `<FUN>` or `<CMD>` tokens needed.

### Config Setup

```python
# Config.py
DATASET_JSON_PATH = "path/to/fun_commands/"  # Directory path
DATASET_FORMAT = "FUN"
image_dir = "path/to/images/"
```

### Train

```bash
python train_florence_new.py
```

## Features

✅ **Directory loading**: One JSON file per sample  
✅ **Plain text commands**: No special tokens required  
✅ **Auto-detection**: Automatically detects format  
✅ **Backward compatible**: Old formats still work

## See Full Documentation

- [Directory Loading Guide](file:///d:/Desktop/Samsung/Y26/train_florence/DIRECTORY_LOADING_GUIDE.md)
- [Full README](file:///d:/Desktop/Samsung/Y26/train_florence/README_DATA_PROCESSORS.md)
