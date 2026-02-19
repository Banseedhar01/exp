# Florence-2 Multi-Dataset Training

Fine-tune Florence-2 on any combination of four dataset types using `accelerate launch`.

---

## Dataset Types

| Flag | Type | Input Format |
|------|------|-------------|
| `--use-action` | ActionDataset | CSV (`action.csv`) |
| `--use-info` | InfoDataset | CSV (`info.csv`) |
| `--use-amex-od` | AMEX Object Detection | JSON |
| `--use-amex-ui` | AMEX UI Action | JSON |

---

## JSON Formats

**AMEX OD:**
```json
{
  "prefix": "<OD>",
  "suffix": "navigate_up<loc_0><loc_28><loc_136><loc_92>...",
  "image":  "filename.png"
}
```

**AMEX UI Action:**
```json
{
  "image":  "filename.png",
  "prefix": "<UI_ACTION> Click to view and book airport transfer services.",
  "suffix": "<loc_525><loc_425><loc_671><loc_492>"
}
```
> `click` is automatically prepended to the suffix: `click <loc_525><loc_425><loc_671><loc_492>`

---

## Quick-Start Examples

### 1 — AMEX UI Action only
```bash
accelerate launch train.py \
  --use-amex-ui \
  --amex-ui-json  /data/amex_ui.json \
  --amex-ui-images /data/amex_ui_images \
  --batch-size 4 --epochs 5 \
  --output-dir ./runs/amex_ui \
  --log-dir    ./runs/amex_ui/logs
```

### 2 — AMEX OD only
```bash
accelerate launch train.py \
  --use-amex-od \
  --amex-od-json   /data/amex_od.json \
  --amex-od-images /data/amex_od_images \
  --batch-size 4 --epochs 5 \
  --output-dir ./runs/amex_od
```

### 3 — Both AMEX datasets combined
```bash
accelerate launch train.py \
  --use-amex-od --amex-od-json /data/od.json --amex-od-images /data/od_imgs \
  --use-amex-ui --amex-ui-json /data/ui.json --amex-ui-images /data/ui_imgs \
  --batch-size 4 --epochs 5 \
  --output-dir ./runs/amex_mixed
```

### 4 — Existing action + info datasets only
```bash
accelerate launch train.py \
  --use-action --action-csv /data/action.csv --action-images /data/imgs \
  --use-info   --info-csv   /data/info.csv \
  --batch-size 2 --epochs 5 \
  --output-dir ./runs/existing
```

### 5 — All four datasets, capped at 10 000 each
```bash
accelerate launch train.py \
  --use-action  --action-csv  /data/action.csv  --action-images /data/imgs \
  --use-info    --info-csv    /data/info.csv \
  --use-amex-od --amex-od-json /data/od.json   --amex-od-images /data/od \
  --use-amex-ui --amex-ui-json /data/ui.json   --amex-ui-images /data/ui \
  --max-action 10000 --max-info 10000 --max-amex-od 10000 --max-amex-ui 10000 \
  --batch-size 4 --epochs 5 \
  --output-dir ./runs/all_mixed
```

### 6 — Test / smoke-test (tiny run)
```bash
python train.py \
  --use-amex-od --amex-od-json /data/od.json --amex-od-images /data/od_imgs \
  --test-mode --test-samples 16 \
  --batch-size 2 --epochs 1
```

---

## Full CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model-path` | `./Florence-2-base` | Path to base model |
| `--output-dir` | `./checkpoints/final` | Checkpoint + final model directory |
| `--log-dir` | `./logs` | Log file directory |
| `--epochs` | `5` | Training epochs |
| `--batch-size` | `2` | Per-GPU batch size |
| `--lr` | `1e-5` | Learning rate |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--max-grad-norm` | `1.0` | Gradient clipping norm |
| `--image-size` | `224` | Resize images to N×N |
| `--no-resize` | — | Disable image resizing |
| `--freeze-vision` | — | Freeze vision encoder weights |
| **Dataset flags** | | |
| `--use-action` | — | Enable ActionDataset (CSV) |
| `--use-info` | — | Enable InfoDataset (CSV) |
| `--use-amex-od` | — | Enable AMEX OD JSON dataset |
| `--use-amex-ui` | — | Enable AMEX UI Action JSON dataset |
| **Dataset paths** | | |
| `--action-csv` | `./data/action.csv` | Action CSV file path |
| `--action-images` | `./data/images` | Image dir for action + info datasets |
| `--info-csv` | `./data/info.csv` | Info CSV file path |
| `--amex-od-json` | `./data/amex_od.json` | AMEX OD annotations JSON |
| `--amex-od-images` | `./data/amex_od_images` | AMEX OD image directory |
| `--amex-ui-json` | `./data/amex_ui_action.json` | AMEX UI Action annotations JSON |
| `--amex-ui-images` | `./data/amex_ui_action_images` | AMEX UI Action image directory |
| **Sample caps** | | |
| `--max-action` | None (all) | Max samples from ActionDataset |
| `--max-info` | None (all) | Max samples from InfoDataset |
| `--max-amex-od` | None (all) | Max samples from AMEX OD dataset |
| `--max-amex-ui` | None (all) | Max samples from AMEX UI dataset |
| **Debug** | | |
| `--test-mode` | — | Use only a small sample slice |
| `--test-samples` | `64` | Size of test slice |

---

## Output Structure

```
<output-dir>/
  checkpoint_epoch_1/
    config.json
    model.safetensors
    preprocessor_config.json
    ...
    meta.json          ← {epoch, avg_loss}
  checkpoint_epoch_2/
  ...
  final/              ← best/last model for inference

<log-dir>/
  training.log
  loss_log_final.txt
  loss_curve_final.png
  loss_curve_epoch_1.png
  ...
```

---

## Notes

- All four datasets are shuffled together via `DistributedSampler(shuffle=True)` — each epoch produces a completely different interleaved order.
- Sample caps (`--max-*`) are applied **before** combining, so you can balance datasets easily.
- The script validates that at least one dataset flag is set and will exit with a clear error if none are provided.
- `train.py` imports `from Config import Config` — the file must be named `Config.py` (capital C) or `config.py` (Python is case-insensitive on Windows).
