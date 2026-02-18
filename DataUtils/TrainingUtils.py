"""
Training Utilities for Florence-2
Provides helper functions for token management, checkpointing, and logging.
"""

import os
import logging
import matplotlib.pyplot as plt
from Config import Config

logger = logging.getLogger(__name__)


def add_custom_tokens(model, processor):
    """
    Add custom task tokens to the Florence-2 model vocabulary.
    Tokens are defined in Config.CUSTOM_TASK_TOKENS.
    e.g. ["<OD>", "<COMMAND>", "<UI_ACTION>", "<CAPTION>", "<EXPECTATION>"]
    """
    task_tokens = Config.CUSTOM_TASK_TOKENS

    # Add as regular tokens first (for embedding lookup)
    new_tokens = [t for t in task_tokens if t not in processor.tokenizer.get_vocab()]
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"Added {len(new_tokens)} new tokens: {new_tokens}")
    else:
        logger.info("All custom tokens already in vocabulary")

    # Also register as special tokens so they are never split by the tokenizer
    processor.tokenizer.add_special_tokens({'additional_special_tokens': task_tokens})

    return model, processor


def plot_loss_curve(losses, save_path="loss_curve.png"):
    """Plot and save the training loss curve."""
    if not losses:
        logger.warning("No losses to plot")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(losses, linewidth=2, color='steelblue')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curve saved to {save_path}")


def save_model_checkpoint(model, processor, epoch, avg_loss, accelerator=None):
    """
    Save a model checkpoint after each epoch.
    Handles both single-GPU and multi-GPU (accelerate) setups.
    """
    checkpoint_dir = f"{Config.model_output_dir}_epoch_{epoch + 1}"
    try:
        logger.info(f"Saving checkpoint to {checkpoint_dir}...")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Unwrap model for multi-GPU training
        unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model

        unwrapped_model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved: {checkpoint_dir} | Loss: {avg_loss:.4f}")

    except Exception as e:
        logger.error(f"Error saving checkpoint at {checkpoint_dir}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def save_loss_log(losses, filename="loss_log.txt"):
    """Save all loss values to a text file."""
    try:
        with open(filename, 'w') as f:
            f.write("Training Loss Log\n")
            f.write("=" * 50 + "\n")
            for i, loss in enumerate(losses):
                f.write(f"Step {i + 1}: {loss:.6f}\n")
        logger.info(f"Loss log saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving loss log: {e}")
