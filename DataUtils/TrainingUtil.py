import os
import json
import logging
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from Config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def add_custom_tokens(model, processor):
    """Add custom task tokens defined in Config.CUSTOM_TASK_TOKENS to the model."""
    task_tokens = Config.CUSTOM_TASK_TOKENS
    processor.tokenizer.add_tokens(task_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    processor.tokenizer.add_special_tokens({'additional_special_tokens': task_tokens})
    return model, processor


# ---------------------------------------------------------------------------
# Loss logging & plotting
# ---------------------------------------------------------------------------

def save_loss_log(losses, filepath):
    """
    Save a list of loss values to a text file with a header and step numbers.

    Args:
        losses:   List[float] of recorded loss values.
        filepath: Destination file path.
    """
    if not losses:
        logger.warning("save_loss_log: no losses to save.")
        return
    try:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write("Training Loss Log\n")
            f.write("=" * 50 + "\n")
            for i, loss in enumerate(losses):
                f.write(f"Step {i + 1}: {loss:.6f}\n")
        logger.info(f"Loss log saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving loss log: {e}")


def plot_loss_curve(losses, filepath):
    """
    Save a loss-vs-step line chart as a PNG file.

    Args:
        losses:   List[float] of recorded loss values.
        filepath: Destination PNG file path.
    """
    if not losses:
        logger.warning("plot_loss_curve: no losses to plot.")
        return
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss curve saved to {filepath}")


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_model_checkpoint(model, processor, epoch, avg_loss, accelerator=None, output_dir="./checkpoints"):
    """
    Save a model checkpoint after each epoch.

    Args:
        model:       The (possibly wrapped) model.
        processor:   The Florence-2 processor.
        epoch:       0-based epoch index.
        avg_loss:    Average loss for this epoch.
        accelerator: Accelerate Accelerator instance (optional for single-GPU).
        output_dir:  Parent directory for checkpoints.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}")
    try:
        logger.info(f"Saving checkpoint to {checkpoint_dir}...")
        os.makedirs(checkpoint_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
        unwrapped.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)

        # Save small metadata JSON alongside the checkpoint
        meta = {"epoch": epoch + 1, "avg_loss": avg_loss}
        with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Checkpoint saved successfully: {checkpoint_dir}, Loss: {avg_loss:.4f}")
    except Exception as e:
        logger.error(f"Error saving checkpoint at {checkpoint_dir}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
