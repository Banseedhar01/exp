"""
Florence-2 Training Script
Supports:
  - Single format training (OD only or COMMAND only)
  - Mixed training (OD + COMMAND simultaneously)
  - Directory-based or single-file datasets

Dataset formats:
  OD:      {"prefix": "<OD>", "suffix": "label<loc_x1>...", "image": "img.png"}
  COMMAND: {"image": "img.png", "prefix": "<COMMAND> Action text.", "suffix": "<loc_x1>..."}
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator

from Config import Config
from DataUtils.unified_dataset import UnifiedFlorenceDataset, MixedFormatDataset

# Optional legacy utilities
try:
    from DataUtils.TrainingUtils import add_custom_tokens, plot_loss_curve, save_model_checkpoint, save_loss_log
except ImportError:
    add_custom_tokens = plot_loss_curve = save_model_checkpoint = save_loss_log = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Dataset builder
# -----------------------------------------------------------------------

def build_dataset() -> torch.utils.data.Dataset:
    """
    Build the training dataset from Config.
    Supports three modes:
      1. Mixed:   OD_DATASET_PATH + COMMAND_DATASET_PATH both set → MixedFormatDataset
      2. OD only: OD_DATASET_PATH set
      3. COMMAND only: COMMAND_DATASET_PATH set
      4. Shortcut: DATASET_JSON_PATH set → UnifiedFlorenceDataset with DATASET_FORMAT

    Each format uses its own image directory (OD_IMAGE_DIR / COMMAND_IMAGE_DIR).
    """
    size_cfg = dict(image_size=Config.IMAGE_SIZE, resize_images=Config.RESIZE_IMAGES)

    # --- Mode 1: Mixed training ---
    if Config.use_mixed_training():
        logger.info("Mixed training mode: OD + COMMAND")
        od_ds = UnifiedFlorenceDataset(
            data_path=Config.OD_DATASET_PATH,
            image_dir=Config.OD_IMAGE_DIR,
            format_type="OD",
            **size_cfg
        )
        cmd_ds = UnifiedFlorenceDataset(
            data_path=Config.COMMAND_DATASET_PATH,
            image_dir=Config.COMMAND_IMAGE_DIR,
            format_type="COMMAND",
            **size_cfg
        )
        return MixedFormatDataset([od_ds, cmd_ds])

    # --- Mode 2: OD only ---
    if Config.OD_DATASET_PATH is not None:
        logger.info("Single-format training: OD")
        return UnifiedFlorenceDataset(
            data_path=Config.OD_DATASET_PATH,
            image_dir=Config.OD_IMAGE_DIR,
            format_type="OD",
            **size_cfg
        )

    # --- Mode 3: COMMAND only ---
    if Config.COMMAND_DATASET_PATH is not None:
        logger.info("Single-format training: COMMAND")
        return UnifiedFlorenceDataset(
            data_path=Config.COMMAND_DATASET_PATH,
            image_dir=Config.COMMAND_IMAGE_DIR,
            format_type="COMMAND",
            **size_cfg
        )

    # --- Mode 4: Shortcut DATASET_JSON_PATH (uses legacy image_dir) ---
    if Config.DATASET_JSON_PATH is not None:
        logger.info(f"Single-format training via DATASET_JSON_PATH (format={Config.DATASET_FORMAT})")
        return UnifiedFlorenceDataset(
            data_path=Config.DATASET_JSON_PATH,
            image_dir=Config.image_dir,
            format_type=Config.DATASET_FORMAT,
            **size_cfg
        )

    raise ValueError(
        "No dataset path configured. Set OD_DATASET_PATH, COMMAND_DATASET_PATH, or DATASET_JSON_PATH in Config."
    )


# -----------------------------------------------------------------------
# Collate
# -----------------------------------------------------------------------

def collate_batch(batch, processor, device, dataset):
    """
    Collate a batch of (prefix, suffix, image_id) tuples.
    Works for OD, COMMAND, and mixed batches — the prefix is passed
    directly to the Florence-2 processor as the task prompt.
    """
    prefixes, suffixes, image_ids = zip(*batch)

    # Load images — for MixedFormatDataset we need the right sub-dataset
    images = []
    for i, image_id in enumerate(image_ids):
        if isinstance(dataset, MixedFormatDataset):
            # Find the sub-dataset that owns this sample
            global_idx = i  # approximate; image_id is unique enough
            sub_ds = dataset.get_dataset_for_index(i)
            images.append(sub_ds.load_image(image_id))
        else:
            images.append(dataset.load_image(image_id))

    # Encode inputs (prefix = task prompt for Florence-2)
    inputs = processor(
        text=list(prefixes),
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Encode labels (suffix = expected output)
    labels = processor.tokenizer(
        text=list(suffixes),
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False
    ).input_ids.to(device)

    return inputs, labels, image_ids


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, lr_scheduler, accelerator,
                epoch, processor, dataset, epoch_losses):
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{Config.EPOCHS}",
        disable=not accelerator.is_main_process
    )

    for batch_idx, batch in enumerate(progress_bar):
        try:
            inputs, labels, image_ids = collate_batch(batch, processor, accelerator.device, dataset)

            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                labels=labels
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

            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'avg': f"{total_loss / num_batches:.4f}",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })

            if batch_idx % 100 == 0 and batch_idx > 0 and accelerator.is_main_process:
                epoch_losses.append(current_loss)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM at batch {batch_idx}, skipping")
                torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            logger.error(f"Batch {batch_idx} error: {e}")
            continue

    torch.cuda.empty_cache()
    return total_loss / num_batches if num_batches > 0 else 0.0


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("Florence-2 Training")
        logger.info(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")
        logger.info("=" * 60)

    # ---- Load model & processor ----
    model = AutoModelForCausalLM.from_pretrained("./Florence-2-base", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("./Florence-2-base", trust_remote_code=True)

    # Add custom tokens (<OD>, <COMMAND>, etc.)
    if add_custom_tokens:
        model, processor = add_custom_tokens(model, processor)
    else:
        # Inline token addition if TrainingUtils not available
        new_tokens = [t for t in Config.CUSTOM_TASK_TOKENS
                      if t not in processor.tokenizer.get_vocab()]
        if new_tokens:
            processor.tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(processor.tokenizer))
            if accelerator.is_main_process:
                logger.info(f"Added {len(new_tokens)} custom tokens: {new_tokens}")

    if Config.FREEZE_VISION_ENCODER:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        if accelerator.is_main_process:
            logger.info("Vision encoder frozen")

    # ---- Build dataset ----
    if accelerator.is_main_process:
        logger.info("Building dataset...")

    dataset = build_dataset()

    if accelerator.is_main_process:
        logger.info(f"Total samples: {len(dataset)}")
        if isinstance(dataset, MixedFormatDataset):
            logger.info("Training on mixed OD + COMMAND data")
        elif isinstance(dataset, UnifiedFlorenceDataset):
            logger.info(f"Training on {dataset.get_format_name()} data")

    # ---- DataLoader ----
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True
    )
    train_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=lambda batch: batch  # raw tuples; collated in collate_batch
    )

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    num_steps = Config.EPOCHS * len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
    num_warmup = int(Config.WARMUP_RATIO * num_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_steps)

    if accelerator.is_main_process:
        logger.info(f"Training steps: {num_steps} | Warmup: {num_warmup}")

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # ---- Training ----
    all_losses = []
    try:
        for epoch in range(Config.EPOCHS):
            train_sampler.set_epoch(epoch)
            epoch_losses = []
            avg_loss = train_epoch(
                model, train_loader, optimizer, lr_scheduler,
                accelerator, epoch, processor, dataset, epoch_losses
            )
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                all_losses.extend(epoch_losses)
                logger.info(f"Epoch {epoch + 1}/{Config.EPOCHS} — avg loss: {avg_loss:.4f}")

                if save_model_checkpoint:
                    save_model_checkpoint(model, processor, epoch, avg_loss, accelerator)
                if plot_loss_curve and all_losses:
                    plot_loss_curve(all_losses, f"loss_curve_epoch_{epoch + 1}.png")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ---- Save final model ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(Config.model_output_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(Config.model_output_dir)
        processor.save_pretrained(Config.model_output_dir)

        if save_loss_log:
            save_loss_log(all_losses, "loss_log_final.txt")
        if plot_loss_curve and all_losses:
            plot_loss_curve(all_losses, "loss_curve_final.png")

        logger.info("=" * 60)
        logger.info(f"Training complete. Model saved to: {Config.model_output_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
