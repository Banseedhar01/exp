"""
train.py — Florence-2 multi-GPU training script with multi-dataset support.

Supports four dataset types:
  - action       : existing ActionDataset  (CSV)
  - info         : existing InfoDataset    (CSV)
  - amex_od      : AMEX Object Detection   (JSON)
  - amex_ui      : AMEX UI Action          (JSON)

Any combination can be enabled via CLI flags.  Datasets are concatenated and
shuffled together via DistributedSampler so multi-task batches are well-mixed.

Example usage:
    # Train on AMEX OD + UI Action only
    accelerate launch train.py \\
        --use-amex-od --amex-od-json /data/od.json --amex-od-images /data/od_imgs \\
        --use-amex-ui --amex-ui-json /data/ui.json --amex-ui-images /data/ui_imgs \\
        --batch-size 4 --epochs 3 --output-dir ./runs/amex_mixed

    # Train on all four datasets, cap each at 10 000 samples
    accelerate launch train.py \\
        --use-action  --action-csv /data/action.csv  --action-images /data/imgs \\
        --use-info    --info-csv   /data/info.csv                                \\
        --use-amex-od --amex-od-json /data/od.json   --amex-od-images /data/od  \\
        --use-amex-ui --amex-ui-json /data/ui.json   --amex-ui-images /data/ui  \\
        --max-action 10000 --max-info 10000 --max-amex-od 10000 --max-amex-ui 10000
"""

import os
import sys
import argparse
import logging
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from accelerate import Accelerator

# Config is imported first; CLI args will patch its class attributes
from Config import Config

from DataUtils.TrainingUtil import (
    add_custom_tokens,
    plot_loss_curve,
    save_model_checkpoint,
    save_loss_log,
)


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Florence-2 multi-dataset training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model --
    parser.add_argument("--model-path",    default=Config.model_path,    help="Path to Florence-2 base model")
    parser.add_argument("--freeze-vision", action="store_true",           help="Freeze vision encoder weights")

    # -- Output / Logging --
    parser.add_argument("--output-dir", default=Config.model_output_dir, help="Directory for saved checkpoints and final model")
    parser.add_argument("--log-dir",    default=Config.log_dir,           help="Directory for log files")

    # -- Training hyper-parameters --
    parser.add_argument("--epochs",      type=int,   default=Config.EPOCHS,                      help="Number of training epochs")
    parser.add_argument("--batch-size",  type=int,   default=Config.BATCH_SIZE,                  help="Per-GPU batch size")
    parser.add_argument("--lr",          type=float, default=Config.LEARNING_RATE,               help="Learning rate")
    parser.add_argument("--grad-accum",  type=int,   default=Config.GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=Config.MAX_GRAD_NORM,             help="Gradient clipping norm")

    # -- Image --
    parser.add_argument("--image-size",     type=int,         default=Config.IMAGE_SIZE,   help="Resize images to this size")
    parser.add_argument("--no-resize",      action="store_true",                           help="Disable image resizing")

    # -- Dataset enable flags --
    parser.add_argument("--use-action",   action="store_true", help="Enable existing ActionDataset (CSV)")
    parser.add_argument("--use-info",     action="store_true", help="Enable existing InfoDataset (CSV)")
    parser.add_argument("--use-amex-od",  action="store_true", help="Enable AMEX OD JSON dataset")
    parser.add_argument("--use-amex-ui",  action="store_true", help="Enable AMEX UI Action JSON dataset")

    # -- Dataset paths --
    parser.add_argument("--action-csv",    default=Config.commands_path,         help="Path to action CSV file")
    parser.add_argument("--action-images", default=Config.image_dir,             help="Image directory for action/info datasets")
    parser.add_argument("--info-csv",      default=Config.info_path,             help="Path to info CSV file")
    parser.add_argument("--amex-od-json",    default=Config.amex_od_json,        help="Path to AMEX OD annotations JSON")
    parser.add_argument("--amex-od-images",  default=Config.amex_od_image_dir,   help="Image directory for AMEX OD dataset")
    parser.add_argument("--amex-ui-json",    default=Config.amex_ui_action_json, help="Path to AMEX UI Action annotations JSON")
    parser.add_argument("--amex-ui-images",  default=Config.amex_ui_action_image_dir, help="Image directory for AMEX UI Action dataset")

    # -- Per-dataset sample caps --
    parser.add_argument("--max-action",   type=int, default=None, help="Max samples from ActionDataset (default: all)")
    parser.add_argument("--max-info",     type=int, default=None, help="Max samples from InfoDataset (default: all)")
    parser.add_argument("--max-amex-od",  type=int, default=None, help="Max samples from AMEX OD dataset (default: all)")
    parser.add_argument("--max-amex-ui",  type=int, default=None, help="Max samples from AMEX UI Action dataset (default: all)")

    # -- Test / debug --
    parser.add_argument("--test-mode",    action="store_true",              help="Enable test mode (small dataset slice)")
    parser.add_argument("--test-samples", type=int, default=Config.TEST_SAMPLE_SIZE, help="Number of samples in test mode")

    return parser.parse_args()


def apply_args_to_config(args):
    """Patch Config class attributes with parsed CLI values."""
    Config.model_path                = args.model_path
    Config.FREEZE_VISION_ENCODER     = args.freeze_vision
    Config.model_output_dir          = args.output_dir
    Config.log_dir                   = args.log_dir
    Config.EPOCHS                    = args.epochs
    Config.BATCH_SIZE                = args.batch_size
    Config.LEARNING_RATE             = args.lr
    Config.GRADIENT_ACCUMULATION_STEPS = args.grad_accum
    Config.MAX_GRAD_NORM             = args.max_grad_norm
    Config.IMAGE_SIZE                = args.image_size
    Config.RESIZE_IMAGES             = not args.no_resize
    Config.USE_ACTION                = args.use_action
    Config.USE_INFO                  = args.use_info
    Config.USE_AMEX_OD               = args.use_amex_od
    Config.USE_AMEX_UI_ACTION        = args.use_amex_ui
    Config.commands_path             = args.action_csv
    Config.image_dir                 = args.action_images
    Config.info_path                 = args.info_csv
    Config.amex_od_json              = args.amex_od_json
    Config.amex_od_image_dir         = args.amex_od_images
    Config.amex_ui_action_json       = args.amex_ui_json
    Config.amex_ui_action_image_dir  = args.amex_ui_images
    Config.MAX_ACTION                = args.max_action
    Config.MAX_INFO                  = args.max_info
    Config.MAX_AMEX_OD               = args.max_amex_od
    Config.MAX_AMEX_UI_ACTION        = args.max_amex_ui
    Config.TEST_MODE                 = args.test_mode
    Config.TEST_SAMPLE_SIZE          = args.test_samples


# ============================================================
# Logging setup
# ============================================================

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)
    return logger


# ============================================================
# Mixed Florence Dataset
# ============================================================

class MixedFlorenceDataset(Dataset):
    """
    A unified PyTorch Dataset that holds items from multiple source datasets.
    Each item carries its own image-loader function so no global image_dir is needed.

    Item format returned by __getitem__:
        (prefix, suffix, image_id, image_loader_fn)
    """

    def __init__(self, datasets_with_loaders):
        """
        Args:
            datasets_with_loaders: list of (items_list, image_loader_fn) tuples.
                items_list  — list of dicts: [{prefix, suffix, image_id}, ...]
                image_loader_fn — callable(image_id) -> PIL.Image
        """
        self._data = []
        for items, loader_fn in datasets_with_loaders:
            for item in items:
                self._data.append((item["prefix"], item["suffix"], item["image_id"], loader_fn))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


# ============================================================
# Dataset loading
# ============================================================

def _cap(data_list, max_samples):
    """Return data_list[:max_samples] if max_samples is set, else full list."""
    if max_samples is not None and max_samples > 0:
        return data_list[:max_samples]
    return data_list


def load_datasets(logger):
    """
    Load all enabled datasets, apply sample caps, print a count summary,
    and return a single MixedFlorenceDataset + a {name: count} summary dict.
    """
    datasets_with_loaders = []
    counts = {}

    # ---- ActionDataset -------------------------------------------------------
    if Config.USE_ACTION:
        from DataUtils.ActionDataset import ActionDataset
        logger.info("[Loader] Loading ActionDataset …")
        ds = ActionDataset(Config.commands_path)
        # Patch image_dir so load_image uses CLI-provided path
        ds.dataset_path = Config.commands_path
        Config.image_dir = Config.image_dir  # already set

        raw = ds.getData()
        capped = _cap(raw, Config.MAX_ACTION)
        if Config.TEST_MODE:
            capped = capped[: Config.TEST_SAMPLE_SIZE]

        # ActionDataset.load_image uses Config.image_dir internally; wrap it
        import os as _os
        from PIL import Image as _Img

        _action_image_dir = Config.image_dir

        def action_loader(image_id,
                          _dir=_action_image_dir,
                          _resize=Config.RESIZE_IMAGES,
                          _sz=Config.IMAGE_SIZE):
            path = _os.path.join(_dir, f"{image_id}.png")
            try:
                img = _Img.open(path).convert("RGB")
                if _resize:
                    img = img.resize((_sz, _sz), _Img.Resampling.LANCZOS)
                return img
            except Exception as e:
                logger.warning(f"action_loader: {e}")
                return _Img.new("RGB", (_sz, _sz), color="white")

        datasets_with_loaders.append((capped, action_loader))
        counts["action"] = len(capped)
        logger.info(f"[Loader] ActionDataset  → {len(capped):,} samples")

    # ---- InfoDataset ---------------------------------------------------------
    if Config.USE_INFO:
        from DataUtils.InfoDataset import InfoDataset
        logger.info("[Loader] Loading InfoDataset …")
        ds = InfoDataset(Config.info_path)
        raw = ds.getData()
        capped = _cap(raw, Config.MAX_INFO)
        if Config.TEST_MODE:
            capped = capped[: Config.TEST_SAMPLE_SIZE]

        import os as _os
        from PIL import Image as _Img

        _info_image_dir = Config.image_dir

        def info_loader(image_id,
                        _dir=_info_image_dir,
                        _resize=Config.RESIZE_IMAGES,
                        _sz=Config.IMAGE_SIZE):
            path = _os.path.join(_dir, f"{image_id}.png")
            try:
                img = _Img.open(path).convert("RGB")
                if _resize:
                    img = img.resize((_sz, _sz), _Img.Resampling.LANCZOS)
                return img
            except Exception as e:
                logger.warning(f"info_loader: {e}")
                return _Img.new("RGB", (_sz, _sz), color="white")

        datasets_with_loaders.append((capped, info_loader))
        counts["info"] = len(capped)
        logger.info(f"[Loader] InfoDataset    → {len(capped):,} samples")

    # ---- AMEX OD Dataset -----------------------------------------------------
    if Config.USE_AMEX_OD:
        from DataUtils.AmexODDataset import AmexODDataset
        logger.info("[Loader] Loading AMEX OD Dataset …")
        ds = AmexODDataset(
            json_path=Config.amex_od_json,
            image_dir=Config.amex_od_image_dir,
            max_samples=Config.MAX_AMEX_OD,
        )
        raw = ds.getData()
        if Config.TEST_MODE:
            raw = raw[: Config.TEST_SAMPLE_SIZE]

        datasets_with_loaders.append((raw, ds.load_image))
        counts["amex_od"] = len(raw)
        logger.info(f"[Loader] AMEX OD        → {len(raw):,} samples")

    # ---- AMEX UI Action Dataset ----------------------------------------------
    if Config.USE_AMEX_UI_ACTION:
        from DataUtils.AmexUIActionDataset import AmexUIActionDataset
        logger.info("[Loader] Loading AMEX UI Action Dataset …")
        ds = AmexUIActionDataset(
            json_path=Config.amex_ui_action_json,
            image_dir=Config.amex_ui_action_image_dir,
            max_samples=Config.MAX_AMEX_UI_ACTION,
        )
        raw = ds.getData()
        if Config.TEST_MODE:
            raw = raw[: Config.TEST_SAMPLE_SIZE]

        datasets_with_loaders.append((raw, ds.load_image))
        counts["amex_ui"] = len(raw)
        logger.info(f"[Loader] AMEX UI Action → {len(raw):,} samples")

    if not datasets_with_loaders:
        raise ValueError(
            "No datasets enabled! Use at least one of: "
            "--use-action, --use-info, --use-amex-od, --use-amex-ui"
        )

    mixed = MixedFlorenceDataset(datasets_with_loaders)
    return mixed, counts


def print_dataset_summary(counts, logger):
    logger.info("=" * 50)
    logger.info("[Dataset Summary]")
    max_name_len = max(len(k) for k in counts)
    for name, count in counts.items():
        logger.info(f"  {name:<{max_name_len + 2}}: {count:>10,} samples")
    logger.info(f"  {'TOTAL':<{max_name_len + 2}}: {sum(counts.values()):>10,} samples")
    logger.info("=" * 50)


# ============================================================
# Collate & Training
# ============================================================

def collate_batch(batch, processor, device, logger=None, debug=False):
    """
    Collate a list of (prefix, suffix, image_id, image_loader_fn) tuples
    into model inputs and labels.
    """
    prefixes, suffixes, image_ids, loaders = zip(*batch)

    # Load images using each item's own loader
    images = [loader(img_id) for loader, img_id in zip(loaders, image_ids)]

    if debug and logger:
        logger.info(f"[collate_batch] Batch size: {len(prefixes)}")
        logger.info(f"[collate_batch] Sample prefix: {prefixes[0][:80]}")
        logger.info(f"[collate_batch] Sample suffix: {suffixes[0][:80]}")

    inputs = processor(
        text=list(prefixes),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    labels = processor.tokenizer(
        text=list(suffixes),
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
    ).input_ids.to(device)

    return inputs, labels, image_ids


def train_epoch(model, train_loader, optimizer, lr_scheduler, accelerator,
                epoch, processor, epoch_losses, global_step, logger):
    model.train()
    total_loss = 0.0
    num_batches = 0
    samples_processed = 0

    if accelerator.is_main_process:
        logger.info(f"Epoch {epoch + 1}/{Config.EPOCHS} — {len(train_loader)} batches")

    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_main_process)

    for batch_idx, batch in enumerate(progress):
        is_first = batch_idx == 0
        try:
            inputs, labels, image_ids = collate_batch(
                batch, processor, accelerator.device,
                logger=logger if accelerator.is_main_process else None,
                debug=is_first and accelerator.is_main_process,
            )

            samples_processed += len(image_ids)

            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                labels=labels,
            )
            loss = outputs.loss

            if Config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            current_loss = loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
            total_loss += current_loss
            num_batches += 1

            current_lr = lr_scheduler.get_last_lr()[0]
            progress.set_postfix({
                "loss": f"{current_loss:.4f}",
                "avg":  f"{total_loss / num_batches:.4f}",
                "lr":   f"{current_lr:.2e}",
            })

            if batch_idx % 100 == 0 and batch_idx > 0 and accelerator.is_main_process:
                epoch_losses.append(current_loss)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if accelerator.is_local_main_process:
                    logger.error(f"OOM at batch {batch_idx} — skipping")
                torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            if accelerator.is_local_main_process:
                import traceback
                logger.error(f"Batch {batch_idx} error: {e}\n{traceback.format_exc()}")
            continue

    if accelerator.is_main_process:
        if num_batches > 0:
            avg = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1} done — avg_loss={avg:.4f} batches={num_batches} samples_this_gpu={samples_processed}")
        else:
            logger.warning(f"Epoch {epoch + 1} — 0 batches processed")

    torch.cuda.empty_cache()
    return total_loss / num_batches if num_batches > 0 else 0.0


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    apply_args_to_config(args)

    # Directories
    os.makedirs(Config.log_dir, exist_ok=True)
    os.makedirs(Config.model_output_dir, exist_ok=True)

    logger = setup_logging(Config.log_dir)

    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info("=" * 50)
        logger.info("Florence-2 Multi-Dataset Training")
        logger.info("=" * 50)
        logger.info(f"Device: {accelerator.device}  |  Processes: {accelerator.num_processes}")
        logger.info(f"Batch size: {Config.BATCH_SIZE}  |  Epochs: {Config.EPOCHS}  |  LR: {Config.LEARNING_RATE}")
        logger.info(f"Output: {Config.model_output_dir}  |  Logs: {Config.log_dir}")

    # ---- Model & Processor ---------------------------------------------------
    if accelerator.is_main_process:
        logger.info(f"Loading model from {Config.model_path} …")

    model = AutoModelForCausalLM.from_pretrained(Config.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(Config.model_path, trust_remote_code=True)
    model, processor = add_custom_tokens(model, processor)

    if Config.FREEZE_VISION_ENCODER:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        if accelerator.is_main_process:
            logger.info("Vision encoder frozen.")

    # ---- Datasets ------------------------------------------------------------
    if accelerator.is_main_process:
        logger.info("Loading datasets …")

    mixed_dataset, counts = load_datasets(logger)

    if accelerator.is_main_process:
        print_dataset_summary(counts, logger)

    # ---- DataLoader (with DistributedSampler for proper shuffle) -------------
    train_sampler = DistributedSampler(
        mixed_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,           # shuffles across ALL combined samples each epoch
    )

    train_loader = DataLoader(
        mixed_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=lambda batch: batch,  # pass raw tuples; collate_batch handles them
        num_workers=0,
        pin_memory=False,
    )

    if accelerator.is_main_process:
        logger.info(f"Total samples: {len(mixed_dataset):,} | Batches/GPU: {len(train_loader)}")

    # ---- Optimizer & Scheduler -----------------------------------------------
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    total_steps = Config.EPOCHS * len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if accelerator.is_main_process:
        logger.info(f"Optimizer: AdamW | total_steps={total_steps} | warmup={warmup_steps}")

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # ---- Training loop -------------------------------------------------------
    all_losses = []
    global_step = 0

    try:
        for epoch in range(Config.EPOCHS):
            train_sampler.set_epoch(epoch)   # ensures different shuffle each epoch

            epoch_losses = []
            avg_loss = train_epoch(
                model, train_loader, optimizer, lr_scheduler,
                accelerator, epoch, processor, epoch_losses, global_step, logger,
            )
            global_step += len(train_loader)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                all_losses.extend(epoch_losses)
                ckpt_dir = os.path.join(Config.model_output_dir, f"checkpoint_epoch_{epoch + 1}")
                save_model_checkpoint(model, processor, epoch, avg_loss, accelerator, output_dir=Config.model_output_dir)
                if all_losses:
                    curve_path = os.path.join(Config.log_dir, f"loss_curve_epoch_{epoch + 1}.png")
                    plot_loss_curve(all_losses, curve_path)
                logger.info(f"Checkpoint saved — epoch {epoch + 1}")

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logger.info("Training interrupted by user")
            save_loss_log(all_losses, os.path.join(Config.log_dir, "loss_log_interrupted.txt"))
            if all_losses:
                plot_loss_curve(all_losses, os.path.join(Config.log_dir, "loss_curve_interrupted.png"))
        return

    except Exception as e:
        if accelerator.is_main_process:
            import traceback
            logger.error(f"Training failed: {e}\n{traceback.format_exc()}")
            save_loss_log(all_losses, os.path.join(Config.log_dir, "loss_log_error.txt"))
            if all_losses:
                plot_loss_curve(all_losses, os.path.join(Config.log_dir, "loss_curve_error.png"))
        return

    # ---- Final model save ----------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(Config.model_output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        logger.info(f"Saving final model to {final_dir} …")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        save_loss_log(all_losses, os.path.join(Config.log_dir, "loss_log_final.txt"))
        if all_losses:
            plot_loss_curve(all_losses, os.path.join(Config.log_dir, "loss_curve_final.png"))
        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()