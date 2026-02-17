"""
Qwen LLM UI Element Annotation Generator
=========================================

This script processes JSON files containing UI element annotations and generates:
1. 10 different variations of functionality descriptions
2. Purpose descriptions (10-20 words) for each UI element

Features:
- Multi-GPU support (optimized for 4 V100 GPUs)
- Batch processing with memory optimization
- Comprehensive logging
- OOM recovery and retry logic
- Modular architecture

Author: Generated for Samsung Y26 Project
Date: 2026-02-17
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# =========================
# Configuration
# =========================

@dataclass
class Config:
    """Configuration for Qwen UI element annotation generation."""
    
    # Model configuration
    model_path: str = "/group-volume/SRIB-Bixby-Screen-AI/k.anup/Qwen-models/Qwen3-30B-A3B-Instruct-2507"
    
    # Dataset paths
    input_dir: Path = Path("./input_annotations")
    output_dir: Path = Path("./output_annotations")
    
    # Logging
    log_file: str = "qwen_generation.log"
    log_dir: Path = Path("./logs")
    
    # Generation parameters
    num_variations: int = 10  # Number of functionality variations
    max_new_tokens: int = 200
    temperature: float = 0.7  # Higher for more variation
    top_p: float = 0.9
    top_k: int = 50
    
    # Batch processing
    batch_size: int = 8  # Elements per batch
    min_batch_size: int = 1
    
    # GPU configuration
    num_gpus: int = 4
    device_map: str = "auto"  # Auto distribute across GPUs
    
    # Memory optimization
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"  # or "float16"
    low_cpu_mem_usage: bool = True
    
    # Memory management
    clear_cache_interval: int = 10  # Clear cache every N files
    aggressive_cleanup: bool = True
    min_free_memory_mb: float = 2000.0
    
    # Processing control
    skip_existing: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Testing
    test_mode: bool = False
    test_samples: int = 5
    limit: Optional[int] = None
    verbose: bool = False
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.input_dir.exists():
            errors.append(f"Input directory not found: {self.input_dir}")
        
        if self.num_variations < 1 or self.num_variations > 20:
            errors.append("num_variations must be between 1 and 20")
        
        if self.max_new_tokens <= 0:
            errors.append("max_new_tokens must be positive")
        
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        
        if not 0 <= self.top_p <= 1:
            errors.append("top_p must be between 0 and 1")
        
        if self.batch_size < 1:
            errors.append("batch_size must be at least 1")
        
        if self.num_gpus < 1:
            errors.append("num_gpus must be at least 1")
        
        return errors


# =========================
# Prompts
# =========================

SYSTEM_PROMPT = """You are an expert UI/UX analyst specialized in analyzing mobile and web user interfaces.

Your task is to analyze UI elements and provide detailed, accurate descriptions based on their metadata.

You will be given information about a UI element including:
- Element description (xml_desc)
- Element type
- Element index
- Page context

Your responses must be:
- Specific and precise
- Based on the provided metadata
- Professional and clear
- Concise but informative"""


def build_variation_prompt(element: Dict[str, Any], variation_num: int) -> str:
    """Build prompt for generating functionality variation.
    
    Args:
        element: Element data from JSON
        variation_num: Which variation number (1-10)
    
    Returns:
        Formatted prompt string
    """
    xml_desc = element.get("xml_desc", [])
    if isinstance(xml_desc, list):
        xml_desc_str = ", ".join([str(x).strip() for x in xml_desc if x and str(x).strip()])
    else:
        xml_desc_str = str(xml_desc).strip()
    
    element_type = element.get("type", "Unknown")
    idx = element.get("idx", -1)
    original_functionality = element.get("functionality", "").strip()
    
    # Build the prompt with optional reference functionality
    prompt = f"""Analyze this UI element and describe what happens when a user interacts with it.

ELEMENT INFORMATION:
- Description: {xml_desc_str}
- Type: {element_type}
- Index: {idx}"""
    
    # Add reference functionality if available
    if original_functionality:
        prompt += f"""
- Reference Functionality: {original_functionality}"""
    
    prompt += f"""

TASK:
Describe the functionality of this UI element. What happens when the user clicks/taps it?"""
    
    if original_functionality:
        prompt += """
Use the reference functionality as a guide, but rephrase it with different wording and perspective."""
    
    prompt += f"""

Requirements:
- Be specific about the action or outcome
- Consider the element type and description"""
    
    if original_functionality:
        prompt += """
- Maintain the same core meaning as the reference functionality"""
    
    prompt += f"""
- Provide a clear, actionable description
- Keep it concise (10-15 words)
- This is variation #{variation_num}, so provide a slightly different phrasing or perspective

Respond with ONLY the functionality description, no extra text or formatting."""
    
    return prompt


def build_purpose_prompt(element: Dict[str, Any]) -> str:
    """Build prompt for generating purpose description.
    
    Args:
        element: Element data from JSON
    
    Returns:
        Formatted prompt string
    """
    xml_desc = element.get("xml_desc", [])
    if isinstance(xml_desc, list):
        xml_desc_str = ", ".join([str(x).strip() for x in xml_desc if x and str(x).strip()])
    else:
        xml_desc_str = str(xml_desc).strip()
    
    element_type = element.get("type", "Unknown")
    idx = element.get("idx", -1)
    
    prompt = f"""Analyze this UI element and describe its purpose in the interface.

ELEMENT INFORMATION:
- Description: {xml_desc_str}
- Type: {element_type}
- Index: {idx}

TASK:
Describe the PURPOSE of this UI element in one clear sentence.

Requirements:
- Explain what this element IS and its role in the interface
- Be specific and precise
- Use 10-20 words
- Focus on the element's function, not the action

Respond with ONLY the purpose description, no extra text or formatting."""
    
    return prompt


# =========================
# GPU Monitor
# =========================

class GPUMonitor:
    """Monitor and manage GPU resources."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.cuda_available = torch.cuda.is_available()
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU memory and utilization info."""
        if not self.cuda_available:
            return {'cuda_available': False, 'gpu_count': 0}
        
        gpu_count = torch.cuda.device_count()
        info = {'cuda_available': True, 'gpu_count': gpu_count}
        
        for idx in range(gpu_count):
            try:
                allocated = torch.cuda.memory_allocated(idx) / (1024**2)
                reserved = torch.cuda.memory_reserved(idx) / (1024**2)
                total = torch.cuda.get_device_properties(idx).total_memory / (1024**2)
                
                info[f"gpu_{idx}"] = {
                    'allocated_mb': allocated,
                    'reserved_mb': reserved,
                    'free_mb': total - reserved,
                    'total_mb': total,
                    'utilization_percent': (reserved / total) * 100
                }
            except Exception as e:
                info[f"gpu_{idx}"] = {'error': str(e)}
        
        return info
    
    def log_gpu_status(self, prefix: str = "GPU Status"):
        """Log current GPU status."""
        info = self.get_gpu_info()
        
        if info['cuda_available']:
            self.logger.info(f"{prefix}:")
            self.logger.info(f"  Device Count: {info['gpu_count']}")
            
            for idx in range(info['gpu_count']):
                gpu_info = info.get(f'gpu_{idx}', {})
                if 'error' not in gpu_info:
                    self.logger.info(
                        f"  GPU {idx}: "
                        f"Allocated: {gpu_info['allocated_mb']:.0f} MB, "
                        f"Reserved: {gpu_info['reserved_mb']:.0f} MB, "
                        f"Free: {gpu_info['free_mb']:.0f} MB"
                    )
        else:
            self.logger.warning(f"{prefix}: CUDA not available")
    
    def clear_cache(self, aggressive: bool = False):
        """Clear GPU cache."""
        if self.cuda_available:
            if aggressive:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.debug("GPU cache cleared (aggressive)")
            else:
                torch.cuda.empty_cache()
                self.logger.debug("GPU cache cleared")
    
    def check_memory_available(self, required_mb: float) -> bool:
        """Check if enough GPU memory is available."""
        if not self.cuda_available:
            return True
        
        info = self.get_gpu_info()
        for idx in range(info['gpu_count']):
            gpu_info = info.get(f'gpu_{idx}', {})
            free_mb = gpu_info.get('free_mb', 0)
            if free_mb < required_mb:
                self.logger.warning(
                    f"Low GPU {idx} memory: {free_mb:.0f} MB free, "
                    f"{required_mb:.0f} MB required"
                )
                return False
        return True


# =========================
# Logging Setup
# =========================

def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.log_dir / config.log_file
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if config.verbose else logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =========================
# Model Loading
# =========================

def load_model(config: Config, logger: logging.Logger, gpu_monitor: GPUMonitor) -> Tuple[Any, Any]:
    """Load Qwen model and tokenizer with multi-GPU support.
    
    Args:
        config: Configuration object
        logger: Logger instance
        gpu_monitor: GPU monitor instance
    
    Returns:
        (model, tokenizer) tuple
    """
    logger.info(f"Loading model from: {config.model_path}")
    logger.info(f"Target GPUs: {config.num_gpus}")
    
    # Determine dtype
    if config.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif config.torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    logger.info(f"Using dtype: {dtype}")
    
    # Log GPU status before loading
    gpu_monitor.log_gpu_status("GPU Status Before Model Load")
    
    # Model loading arguments
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": config.device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }
    
    # Add flash attention if supported
    if config.use_flash_attention:
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            else:
                logger.warning("Flash Attention 2 not supported (requires Ampere+ GPU)")
        except Exception as e:
            logger.warning(f"Could not check GPU capability: {e}")
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        
        # Set padding side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Model loaded successfully")
        
        # Log device map
        if hasattr(model, "hf_device_map"):
            logger.info(f"Device map: {model.hf_device_map}")
        
        # Log GPU status after loading
        gpu_monitor.log_gpu_status("GPU Status After Model Load")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# =========================
# Generation Functions
# =========================

def generate_text(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    config: Config,
    logger: logging.Logger,
    gpu_monitor: GPUMonitor,
) -> List[str]:
    """Generate text for a batch of prompts with OOM recovery.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompts: List of prompts
        config: Configuration
        logger: Logger
        gpu_monitor: GPU monitor
    
    Returns:
        List of generated texts
    """
    batch_size = len(prompts)
    
    while batch_size >= config.min_batch_size:
        try:
            # Clear cache before generation
            if config.aggressive_cleanup:
                gpu_monitor.clear_cache(aggressive=True)
            
            # Check memory
            if not gpu_monitor.check_memory_available(config.min_free_memory_mb):
                logger.warning("Low memory detected, performing cleanup")
                gpu_monitor.clear_cache(aggressive=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Tokenize
            current_prompts = prompts[:batch_size]
            inputs = tokenizer(
                current_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to GPU
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[:, input_length:]
            generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            
            # If we reduced batch size, pad with empty strings
            if batch_size < len(prompts):
                generated_texts.extend([""] * (len(prompts) - batch_size))
            
            return generated_texts
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > config.min_batch_size:
                new_batch_size = max(config.min_batch_size, batch_size // 2)
                logger.warning(f"OOM at batch size {batch_size}, reducing to {new_batch_size}")
                batch_size = new_batch_size
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"Generation error: {e}")
                raise
    
    # If we get here, even min batch size failed
    logger.error("Generation failed even with minimum batch size")
    return [""] * len(prompts)


# =========================
# Processing Functions
# =========================

def process_element(
    element: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    config: Config,
    logger: logging.Logger,
    gpu_monitor: GPUMonitor,
) -> Dict[str, Any]:
    """Process a single element to generate variations and purpose.
    
    Args:
        element: Element data from JSON
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Configuration
        logger: Logger
        gpu_monitor: GPU monitor
    
    Returns:
        Dictionary with generated data
    """
    element_idx = element.get("idx", -1)
    
    try:
        # Generate functionality variations
        variation_prompts = [
            build_variation_prompt(element, i+1) 
            for i in range(config.num_variations)
        ]
        
        variations = generate_text(
            model, tokenizer, variation_prompts, config, logger, gpu_monitor
        )
        
        # Clean up variations
        variations = [v.strip() for v in variations]
        
        # Generate purpose
        purpose_prompt = build_purpose_prompt(element)
        purpose_result = generate_text(
            model, tokenizer, [purpose_prompt], config, logger, gpu_monitor
        )
        purpose = purpose_result[0].strip() if purpose_result else ""
        
        return {
            "functionality_variations": variations,
            "purpose": purpose
        }
        
    except Exception as e:
        logger.error(f"Error processing element {element_idx}: {e}")
        return {
            "functionality_variations": ["Error"] * config.num_variations,
            "purpose": "Error"
        }


def process_json_file(
    json_path: Path,
    model: Any,
    tokenizer: Any,
    config: Config,
    logger: logging.Logger,
    gpu_monitor: GPUMonitor,
) -> bool:
    """Process a single JSON file.
    
    Args:
        json_path: Path to input JSON file
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Configuration
        logger: Logger
        gpu_monitor: GPU monitor
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if output exists
        output_path = config.output_dir / json_path.name
        if config.skip_existing and output_path.exists():
            logger.debug(f"Skipping {json_path.name} (already exists)")
            return True
        
        # Load input JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get clickable elements
        elements = data.get("clickable_elements", [])
        if not elements:
            logger.warning(f"No clickable elements in {json_path.name}")
            # Still save the file with no modifications
            config.output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        
        logger.info(f"Processing {json_path.name}: {len(elements)} elements")
        
        # Process each element
        for element in tqdm(elements, desc=f"  Elements", leave=False, disable=not config.verbose):
            generated = process_element(element, model, tokenizer, config, logger, gpu_monitor)
            
            # Append to element
            element["functionality_variations"] = generated["functionality_variations"]
            element["purpose"] = generated["purpose"]
        
        # Save output
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Completed {json_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {json_path.name}: {e}", exc_info=config.verbose)
        return False


# =========================
# Statistics
# =========================

def print_statistics(
    total_files: int,
    successful: int,
    failed: int,
    elapsed_time: float,
    logger: logging.Logger
):
    """Print processing statistics."""
    logger.info("=" * 70)
    logger.info("PROCESSING STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/total_files*100:.1f}%" if total_files > 0 else "N/A")
    logger.info(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    if total_files > 0:
        logger.info(f"Avg time per file: {elapsed_time/total_files:.1f}s")
    logger.info("=" * 70)


# =========================
# Main
# =========================

def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Qwen LLM UI Element Annotation Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    parser.add_argument('--model-path', type=str, help='Path to Qwen model')
    parser.add_argument('--input-dir', type=str, required=True, help='Input JSON directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output JSON directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    
    # Generation parameters
    parser.add_argument('--num-variations', type=int, default=10, help='Number of functionality variations')
    parser.add_argument('--max-new-tokens', type=int, default=200, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    
    # Batch processing
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs to use')
    
    # Memory optimization
    parser.add_argument('--torch-dtype', type=str, default='bfloat16', 
                       choices=['bfloat16', 'float16', 'float32'], help='Torch dtype')
    parser.add_argument('--no-flash-attention', action='store_true', help='Disable flash attention')
    parser.add_argument('--clear-cache-interval', type=int, default=10, 
                       help='Clear cache every N files')
    
    # Processing control
    parser.add_argument('--no-skip-existing', action='store_true', help='Reprocess existing files')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries on failure')
    
    # Testing
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--test-samples', type=int, default=5, help='Number of test samples')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    
    # Apply command line overrides
    if args.model_path:
        config.model_path = args.model_path
    config.input_dir = Path(args.input_dir)
    config.output_dir = Path(args.output_dir)
    config.log_dir = Path(args.log_dir)
    config.num_variations = args.num_variations
    config.max_new_tokens = args.max_new_tokens
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.top_k = args.top_k
    config.batch_size = args.batch_size
    config.num_gpus = args.num_gpus
    config.torch_dtype = args.torch_dtype
    config.use_flash_attention = not args.no_flash_attention
    config.clear_cache_interval = args.clear_cache_interval
    config.skip_existing = not args.no_skip_existing
    config.max_retries = args.max_retries
    config.verbose = args.verbose
    
    if args.test:
        config.test_mode = True
        config.test_samples = args.test_samples
        config.limit = args.test_samples
        config.verbose = True
    
    if args.limit:
        config.limit = args.limit
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("QWEN UI ELEMENT ANNOTATION GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Input dir: {config.input_dir}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Variations: {config.num_variations}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"GPUs: {config.num_gpus}")
    logger.info(f"Test mode: {config.test_mode}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return 1
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(logger)
    
    start_time = time.time()
    
    try:
        # Load model
        model, tokenizer = load_model(config, logger, gpu_monitor)
        
        # Get input files
        json_files = sorted(config.input_dir.glob("*.json"))
        
        if config.limit:
            json_files = json_files[:config.limit]
            logger.info(f"Limiting to {config.limit} files")
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        if not json_files:
            logger.warning("No JSON files found!")
            return 0
        
        # Process files
        successful = 0
        failed = 0
        
        for file_idx, json_path in enumerate(tqdm(json_files, desc="Processing files", unit="file")):
            if process_json_file(json_path, model, tokenizer, config, logger, gpu_monitor):
                successful += 1
            else:
                failed += 1
            
            # Periodic cache clearing
            if (file_idx + 1) % config.clear_cache_interval == 0:
                logger.info(f"Periodic cache clear at file {file_idx + 1}")
                gpu_monitor.clear_cache(aggressive=True)
        
        # Print statistics
        elapsed_time = time.time() - start_time
        print_statistics(len(json_files), successful, failed, elapsed_time, logger)
        
        # Final GPU status
        logger.info("")
        gpu_monitor.log_gpu_status("Final GPU Status")
        
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
