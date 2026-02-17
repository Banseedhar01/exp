"""
Example Training Script for Florence-2 with New Modular Data Processors
Demonstrates how to use the UnifiedFlorenceDataset with both OD and FUN formats
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
from DataUtils.data_validator import DataValidator

# Import legacy utilities if needed
try:
    from DataUtils.TrainingUtils import (
        add_custom_tokens,
        plot_loss_curve,
        save_model_checkpoint,
        save_loss_log
    )
except ImportError:
    print("Warning: Legacy TrainingUtils not found. Some features may be unavailable.")
    add_custom_tokens = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def collate_batch(batch, processor, device, dataset):
    """
    Collate function for the new unified dataset format.
    Works with both OD and FUN formats automatically.
    
    Args:
        batch: List of tuples (prefix, suffix, image_id)
        processor: Florence-2 processor
        device: Target device
        dataset: UnifiedFlorenceDataset instance
    
    Returns:
        Tuple of (inputs, labels, image_ids)
    """
    try:
        # Unpack batch
        prefixes, suffixes, image_ids = zip(*batch)
    except (ValueError, TypeError) as e:
        logger.error(f"Error unpacking batch: {e}")
        raise
    
    # Load images
    images = [dataset.load_image(image_id) for image_id in image_ids]
    
    # Format task inputs - the prefix is already formatted correctly
    task_inputs = [dataset._format_task_input(prefix) for prefix in prefixes]
    
    # Process inputs with Florence-2 processor
    inputs = processor(
        text=task_inputs,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Process labels
    labels = processor.tokenizer(
        text=suffixes,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False
    ).input_ids.to(device)
    
    return inputs, labels, image_ids


def train_epoch(model, train_loader, optimizer, lr_scheduler, accelerator, epoch, 
                processor, dataset, epoch_losses):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    if accelerator.is_main_process:
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}")
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                       disable=not accelerator.is_main_process)
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            inputs, labels, image_ids = collate_batch(
                batch, processor, accelerator.device, dataset
            )
            
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
            
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'avg': f"{total_loss / num_batches:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            if batch_idx % 100 == 0 and batch_idx > 0 and accelerator.is_main_process:
                epoch_losses.append(current_loss)
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM at batch {batch_idx}")
                torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            logger.error(f"Batch {batch_idx} error: {e}")
            continue
    
    torch.cuda.empty_cache()
    return total_loss / num_batches if num_batches > 0 else 0


def main():
    """Main training function."""
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("Florence-2 Training with Modular Data Processors")
        logger.info("=" * 60)
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Processes: {accelerator.num_processes}")
    
    # Validate configuration
    try:
        Config.validate_config()
        if accelerator.is_main_process:
            logger.info("✓ Configuration validated")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Validate dataset if using new format
    if Config.use_new_dataset_format():
        if accelerator.is_main_process:
            logger.info("Validating dataset...")
            results = DataValidator.validate_json_file(
                Config.DATASET_JSON_PATH,
                Config.DATASET_FORMAT
            )
            DataValidator.print_validation_report(results)
            
            if not results['valid']:
                logger.error("Dataset validation failed. Please fix errors before training.")
                return
    
    # Load model and processor
    if accelerator.is_main_process:
        logger.info("Loading model and processor...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "./Florence-2-base",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "./Florence-2-base",
        trust_remote_code=True
    )
    
    # Add custom tokens if function is available
    if add_custom_tokens:
        model, processor = add_custom_tokens(model, processor)
    
    if Config.FREEZE_VISION_ENCODER:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    
    # Create dataset
    if accelerator.is_main_process:
        logger.info("Loading dataset...")
    
    if Config.use_new_dataset_format():
        # Use new unified dataset
        dataset = UnifiedFlorenceDataset(
            data_path=Config.DATASET_JSON_PATH,
            image_dir=Config.image_dir,
            image_size=Config.IMAGE_SIZE,
            resize_images=Config.RESIZE_IMAGES,
            format_type=Config.DATASET_FORMAT
        )
        
        if accelerator.is_main_process:
            logger.info(f"Dataset format: {dataset.get_format_name()}")
            stats = dataset.get_statistics()
            logger.info(f"Dataset statistics: {stats}")
    else:
        # Use legacy dataset (backward compatibility)
        from DataUtils.ActionDataset import ActionDataset, FlorenceActionDataset
        action_dataset = ActionDataset(Config.commands_path)
        dataset = FlorenceActionDataset(action_dataset)
        
        if accelerator.is_main_process:
            logger.info("Using legacy ActionDataset format")
    
    if accelerator.is_main_process:
        logger.info(f"Total samples: {len(dataset)}")
    
    # Create distributed sampler
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
        collate_fn=lambda batch: batch  # Disable automatic collation
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    num_training_steps = Config.EPOCHS * len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(Config.WARMUP_RATIO * num_training_steps)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    if accelerator.is_main_process:
        logger.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        logger.info("=" * 60)
    
    # Prepare for distributed training
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    
    # Training loop
    all_losses = []
    
    try:
        for epoch in range(Config.EPOCHS):
            train_sampler.set_epoch(epoch)
            
            epoch_losses = []
            avg_loss = train_epoch(
                model, train_loader, optimizer, lr_scheduler, accelerator,
                epoch, processor, dataset, epoch_losses
            )
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                all_losses.extend(epoch_losses)
                
                # Save checkpoint if function is available
                if save_model_checkpoint:
                    save_model_checkpoint(model, processor, epoch, avg_loss, accelerator)
                
                if plot_loss_curve and all_losses:
                    plot_loss_curve(all_losses, f"loss_curve_epoch_{epoch+1}.png")
                
                logger.info(f"Epoch {epoch+1} completed - Loss: {avg_loss:.4f}")
    
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logger.info("Training interrupted")
        return
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return
    
    # Save final model
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        os.makedirs(Config.model_output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(Config.model_output_dir)
        processor.save_pretrained(Config.model_output_dir)
        
        if save_loss_log:
            save_loss_log(all_losses, "loss_log_final.txt")
        if plot_loss_curve and all_losses:
            plot_loss_curve(all_losses, "loss_curve_final.png")
        
        logger.info("=" * 60)
        logger.info("Training completed successfully")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
