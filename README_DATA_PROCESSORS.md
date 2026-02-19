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
# Single GPU — all defaults from Config.py
python train_florence_new.py

# Single GPU — override key params at runtime
python train_florence_new.py \
  --batch-size 4 \
  --grad-accum 2 \
  --epochs 10 \
  --lr 2e-5 \
  --output-dir models/run1 \
  --log-file logs/run1.log

# Multi-GPU with Accelerate (4 GPUs)
accelerate launch --num_processes 4 train_florence_new.py \
  --batch-size 2 \
  --grad-accum 4 \
  --output-dir models/run1 \
  --log-file logs/run1.log
```

#### All CLI Arguments

| Argument | Type | Default (from Config) | Description |
|---|---|---|---|
| `--batch-size` | int | `Config.BATCH_SIZE` | Per-GPU batch size |
| `--grad-accum` | int | `Config.GRADIENT_ACCUMULATION_STEPS` | Gradient accumulation steps |
| `--epochs` | int | `Config.EPOCHS` | Number of training epochs |
| `--lr` | float | `Config.LEARNING_RATE` | Learning rate |
| `--output-dir` | str | `Config.model_output_dir` | Where to save the trained model |
| `--log-file` | str | `Config.LOG_FILE` | Training log file path (auto-creates dir) |

> All args are optional — values from `Config.py` are used as defaults when not specified.

#### Multi-GPU Scaling Guide

Keep the **effective batch size** (`BATCH_SIZE × GRAD_ACCUM × num_GPUs`) consistent across GPU counts to maintain training stability:

| GPUs | `--batch-size` | `--grad-accum` | Effective batch | Suggested `--lr` |
|---|---|---|---|---|
| 1 | 2 | 8 | 16 | `5e-6` |
| 4 | 2 | 4 | 32 | `1e-5` |
| 8 | 2 | 2 | 32 | `1e-5` |
| 8 | 4 | 2 | 64 | `2e-5` |

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

Florence-2 uses normalized coordinates in the range **[0, 999]** — independent of image size or resize.

- `<loc_0>` = 0% of image dimension
- `<loc_999>` = 100% of image dimension
- `<loc_500>` = 50% of image dimension

### Converting pixel → loc token (for dataset creation)

```python
def pixel_to_loc(px, py, orig_w, orig_h):
    x_norm = int((px / orig_w) * 999)
    y_norm = int((py / orig_h) * 999)
    return x_norm, y_norm

# e.g. bbox (100, 200, 300, 400) on a 1080×1920 image
lx1, ly1 = pixel_to_loc(100, 200, 1080, 1920)   # → 92, 104
lx2, ly2 = pixel_to_loc(300, 400, 1080, 1920)   # → 277, 208
suffix = f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"
```

### Post-processing model output → pixel coordinates

```python
import re

def parse_loc_tokens(text):
    return [int(x) for x in re.findall(r'<loc_(\d+)>', text)]

def loc_to_pixel(val, dimension):
    return int(val / 999 * dimension)

# After inference:
output_text = "<loc_245><loc_312><loc_489><loc_601>"
locs = parse_loc_tokens(output_text)  # [245, 312, 489, 601]

orig_w, orig_h = image.size           # original PIL image size

if len(locs) == 4:   # bounding box
    x1 = loc_to_pixel(locs[0], orig_w)
    y1 = loc_to_pixel(locs[1], orig_h)
    x2 = loc_to_pixel(locs[2], orig_w)
    y2 = loc_to_pixel(locs[3], orig_h)
elif len(locs) == 2: # click point
    cx = loc_to_pixel(locs[0], orig_w)
    cy = loc_to_pixel(locs[1], orig_h)
```

> ⚠️ Always convert relative to the **original** image size, not the resized 768×768 training resolution.

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
