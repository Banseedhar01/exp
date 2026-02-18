# Florence-2 Modular Data Processors

A modular, extensible data processing system for Florence-2 training that supports **OD** (Object Detection) and **COMMAND** (Action/Grounding) formats, with directory-based or single-file datasets.

## Supported Formats

### 1. OD Format (Object Detection)
```json
{
  "prefix": "<OD>",
  "suffix": "label<loc_x1><loc_y1><loc_x2><loc_y2>...",
  "image": "image_name.png"
}
```
Multiple objects are concatenated in the suffix: `button<loc_100><loc_200><loc_300><loc_250>icon<loc_400><loc_100><loc_500><loc_200>`

### 2. COMMAND Format (Action / Grounding)
```json
{
  "image": "screenshot.png",
  "prefix": "<COMMAND> Opens the details page for 'Traktor DJ Test Mix'.",
  "suffix": "<loc_24><loc_440><loc_365><loc_597>"
}
```
One action per sample. The `<COMMAND>` token is added to the model vocabulary.

---

## Dataset Structure

Each sample can be a **separate JSON file** in a directory (recommended) or all samples in a **single JSON array file**.

### Directory layout (recommended)
```
your_project/
├── od_data/                  # OD samples — one JSON per image
│   ├── sample001.json
│   └── ...
├── od_images/                # Images for OD dataset
│   ├── screenshot001.png
│   └── ...
├── commands/                 # COMMAND samples — one JSON per image
│   ├── sample001.json
│   └── ...
└── command_images/           # Images for COMMAND dataset
    ├── 2024_4_30_screenshot.png
    └── ...
```

### Single-file layout (also supported)
```
your_project/
├── od_data.json              # Array of OD samples
├── commands.json             # Array of COMMAND samples
└── images/
```

---

## Quick Start

### 1. Prepare Your Dataset

**OD samples** — one file per image or a single array file:
```json
{
  "prefix": "<OD>",
  "suffix": "search_in_mail<loc_39><loc_37><loc_961><loc_101>open_navigation_drawer<loc_49><loc_41><loc_165><loc_97>",
  "image": "2024_5_6_18_8_e0519163c5ff43fc88443fdcda99d4a4-6.png"
}
```

**COMMAND samples** — one file per image or a single array file:
```json
{
  "image": "2024_4_30_20_51_a3ff6f3070264757a5b7cac45ca70a97-7.png",
  "prefix": "<COMMAND> Opens the details page for 'Traktor DJ Test Mix'.",
  "suffix": "<loc_24><loc_440><loc_365><loc_597>"
}
```

### 2. Configure Training

Edit `Config.py`:

```python
class Config:
    # Image directories — each dataset has its own image folder
    OD_IMAGE_DIR      = "data/od_images/"       # images for OD dataset
    COMMAND_IMAGE_DIR = "data/command_images/"  # images for COMMAND dataset

    # --- Option A: Mixed OD + COMMAND training (recommended) ---
    OD_DATASET_PATH      = "data/od_data/"       # directory or .json file
    COMMAND_DATASET_PATH = "data/commands/"       # directory or .json file

    # --- Option B: Single format ---
    # OD_DATASET_PATH      = "data/od_data/"
    # COMMAND_DATASET_PATH = None

    # Tokens added to the Florence-2 vocabulary
    TOKENS = ["UI_ACTION", "CAPTION", "EXPECTATION", "OD", "COMMAND"]
    CUSTOM_TASK_TOKENS = ["<UI_ACTION>", "<CAPTION>", "<EXPECTATION>", "<OD>", "<COMMAND>"]

    BATCH_SIZE = 2
    EPOCHS = 7
    LEARNING_RATE = 1e-5
```

### 3. Run Training

```bash
# Single GPU
python train_florence_new.py

# Multi-GPU with Accelerate
accelerate launch train_florence_new.py
```

---

## Architecture

```
DataUtils/
├── __init__.py              # Package exports
├── base_processor.py        # Abstract base class
├── od_processor.py          # OD format processor
├── fun_processor.py         # COMMAND format processor (CommandDatasetProcessor)
├── unified_dataset.py       # UnifiedFlorenceDataset + MixedFormatDataset
└── data_validator.py        # Validation utilities
```

### Key Components

#### `BaseDatasetProcessor`
Abstract base providing image loading/resizing, bbox validation, and location token parsing.

#### `ODDatasetProcessor`
- Loads OD JSON from file or directory
- Validates `<OD>` prefix and location tokens in suffix
- Extracts labels and bounding boxes

#### `CommandDatasetProcessor` (in `fun_processor.py`)
- Loads COMMAND JSON from file or directory
- Validates `<COMMAND>` prefix and location tokens in suffix
- Extracts command text and single bounding box
- `FUNDatasetProcessor` is kept as an alias for backward compatibility

#### `UnifiedFlorenceDataset`
- Auto-detects OD vs COMMAND format
- Single PyTorch `Dataset` interface for both formats
- Accepts file path or directory path

#### `MixedFormatDataset`
- Combines multiple `UnifiedFlorenceDataset` instances
- Enables joint OD + COMMAND training in one DataLoader

---

## Usage Examples

### Single Format

```python
from DataUtils import UnifiedFlorenceDataset

# COMMAND format from directory
dataset = UnifiedFlorenceDataset(
    data_path="data/commands/",   # directory of JSON files
    image_dir="data/images/",
    format_type="COMMAND"
)

# OD format from single file
od_dataset = UnifiedFlorenceDataset(
    data_path="data/od_data.json",
    image_dir="data/images/",
    format_type="OD"
)

prefix, suffix, image_id = dataset[0]
image = dataset.load_image(image_id)
```

### Mixed OD + COMMAND Training

```python
from DataUtils import UnifiedFlorenceDataset, MixedFormatDataset
from torch.utils.data import DataLoader

od_dataset = UnifiedFlorenceDataset(
    data_path="data/od_data/",
    image_dir="data/images/",
    format_type="OD"
)
cmd_dataset = UnifiedFlorenceDataset(
    data_path="data/commands/",
    image_dir="data/images/",
    format_type="COMMAND"
)

mixed = MixedFormatDataset([od_dataset, cmd_dataset])
train_loader = DataLoader(mixed, batch_size=2, shuffle=True)
```

### Auto-Detection

```python
# Auto-detects based on prefix content
dataset = UnifiedFlorenceDataset(
    data_path="data/my_data/",
    image_dir="data/images/",
    format_type="AUTO"   # detects <OD> vs <COMMAND>
)
print(dataset.get_format_name())  # "OD" or "COMMAND"
```

### Validation

```python
from DataUtils.data_validator import DataValidator

results = DataValidator.validate_json_file("data/commands/", format_type="COMMAND")
DataValidator.print_validation_report(results)
```

Or from the command line:
```bash
python DataUtils/data_validator.py data/commands/ --format COMMAND
```

---

## Coordinate System

Florence-2 uses normalized coordinates in the range [0, 999]:
- `<loc_0>` = 0% of image dimension
- `<loc_999>` = 100% of image dimension

Example: For a 1000×1000 image, `<loc_500>` = pixel 500.

---

## Adding New Formats

1. Create a processor inheriting from `BaseDatasetProcessor`
2. Implement `_load_data()`, `get_item()`, `get_format_name()`
3. Add detection logic to `UnifiedFlorenceDataset._detect_format()`
4. Add the format token to `Config.CUSTOM_TASK_TOKENS`

```python
from DataUtils.base_processor import BaseDatasetProcessor

class MyFormatProcessor(BaseDatasetProcessor):
    def _load_data(self):
        # load your data into self.data
        pass

    def get_item(self, idx):
        item = self.data[idx]
        return item['prefix'], item['suffix'], item['image']

    def get_format_name(self):
        return "MY_FORMAT"
```

---

## Troubleshooting

### Format Not Detected
- OD: `prefix` must start with `<OD>`
- COMMAND: `prefix` must contain `<COMMAND>`
- Use `format_type="OD"` or `format_type="COMMAND"` explicitly if auto-detection fails

### Dataset Validation Fails
Common issues:
- Missing required fields (`prefix`, `suffix`, `image`)
- Invalid coordinate ranges (must be 0–999)
- Invalid bbox (`x1 >= x2` or `y1 >= y2`)
- Missing `<loc_>` tokens in suffix

### Image Not Found
- Check `image_dir` in `Config.py`
- Ensure the `"image"` field in JSON matches the actual filename (including extension)

---

## License

Same as Florence-2 base model.
