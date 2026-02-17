# Qwen UI Element Annotation Generator

A modular Python script for batch processing UI element annotations using Qwen LLM to generate functionality variations and purpose descriptions.

## Features

- **Multi-GPU Support**: Optimized for 4 V100 GPUs with automatic device mapping
- **Batch Processing**: Efficient batch generation with dynamic batch size adjustment
- **OOM Recovery**: Automatic out-of-memory handling with batch size reduction
- **Comprehensive Logging**: Detailed logs with GPU monitoring and progress tracking
- **Memory Optimization**: Flash Attention 2, aggressive cache clearing, and memory monitoring
- **Modular Architecture**: Clean, maintainable code structure

## Requirements

- Python 3.8+
- CUDA-capable GPUs (optimized for 4x V100)
- 20K+ JSON files with UI element annotations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for better performance
pip install flash-attn --no-build-isolation
```

## Input Format

Your JSON files should have this structure:

```json
{
  "page_caption": "Description of the page",
  "image_path": "path/to/screenshot.png",
  "clickable_elements": [
    {
      "bbox": [x1, y1, x2, y2],
      "xml_desc": ["Element description"],
      "type": "Element('Type')",
      "idx": 0,
      "functionality": "Original functionality description"
    }
  ]
}
```

## Output Format

The script appends to each element:

```json
{
  "bbox": [x1, y1, x2, y2],
  "xml_desc": ["Element description"],
  "type": "Element('Type')",
  "idx": 0,
  "functionality": "Original functionality description",
  "functionality_variations": [
    "Variation 1",
    "Variation 2",
    ...
    "Variation 10"
  ],
  "purpose": "10-20 word purpose description"
}
```

## Usage

### Basic Usage

```bash
python qwen_ui_generator.py \
  --input-dir /path/to/input/jsons \
  --output-dir /path/to/output/jsons \
  --model-path /group-volume/SRIB-Bixby-Screen-AI/k.anup/Qwen-models/Qwen3-30B-A3B-Instruct-2507
```

### Full Example with All Options

```bash
python qwen_ui_generator.py \
  --input-dir ./input_annotations \
  --output-dir ./output_annotations \
  --model-path /path/to/qwen/model \
  --num-variations 10 \
  --batch-size 8 \
  --num-gpus 4 \
  --max-new-tokens 200 \
  --temperature 0.7 \
  --top-p 0.9 \
  --torch-dtype bfloat16 \
  --clear-cache-interval 10 \
  --log-dir ./logs \
  --verbose
```

### Test Mode

Run on a small sample to verify everything works:

```bash
python qwen_ui_generator.py \
  --input-dir ./input_annotations \
  --output-dir ./output_annotations \
  --model-path /path/to/qwen/model \
  --test \
  --test-samples 5 \
  --verbose
```

## Command Line Arguments

### Required Arguments

- `--input-dir`: Directory containing input JSON files
- `--output-dir`: Directory to save output JSON files

### Model Configuration

- `--model-path`: Path to Qwen model (default: from config)
- `--num-gpus`: Number of GPUs to use (default: 4)
- `--torch-dtype`: Torch dtype [bfloat16|float16|float32] (default: bfloat16)
- `--no-flash-attention`: Disable flash attention

### Generation Parameters

- `--num-variations`: Number of functionality variations (default: 10)
- `--max-new-tokens`: Maximum tokens to generate (default: 200)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling (default: 0.9)
- `--top-k`: Top-k sampling (default: 50)

### Batch Processing

- `--batch-size`: Initial batch size (default: 8)
- `--clear-cache-interval`: Clear GPU cache every N files (default: 10)

### Processing Control

- `--no-skip-existing`: Reprocess existing output files
- `--max-retries`: Max retries on failure (default: 3)
- `--limit`: Limit number of files to process

### Logging & Testing

- `--log-dir`: Log directory (default: ./logs)
- `--verbose`: Enable verbose logging
- `--test`: Run in test mode
- `--test-samples`: Number of test samples (default: 5)

## Performance Tips

1. **Batch Size**: Start with 8 and adjust based on GPU memory
2. **Flash Attention**: Ensure flash-attn is installed for better performance
3. **Cache Clearing**: Adjust `--clear-cache-interval` based on memory usage
4. **Temperature**: Higher values (0.7-0.9) give more variation diversity

## Monitoring

The script provides:

- Real-time progress bars
- GPU memory monitoring
- Processing statistics
- Detailed logs in `./logs/qwen_generation.log`

### GPU Memory Monitoring

The script automatically monitors GPU memory and:
- Logs memory usage before/after model loading
- Checks available memory before generation
- Performs aggressive cleanup when memory is low
- Reduces batch size on OOM errors

## Troubleshooting

### Out of Memory Errors

1. Reduce `--batch-size` (try 4, 2, or 1)
2. Use `--torch-dtype float16` instead of bfloat16
3. Reduce `--max-new-tokens`
4. Lower `--clear-cache-interval`

### Slow Processing

1. Increase `--batch-size` if memory allows
2. Ensure flash-attention is installed
3. Check GPU utilization with `nvidia-smi`
4. Verify all 4 GPUs are being used (check logs)

### Model Loading Issues

1. Verify model path is correct
2. Check CUDA and PyTorch installation
3. Ensure sufficient disk space for model offloading
4. Try `--no-flash-attention` if flash-attn fails

## Example Workflow

```bash
# 1. Test on small sample
python qwen_ui_generator.py \
  --input-dir ./input_annotations \
  --output-dir ./test_output \
  --model-path /path/to/model \
  --test --test-samples 5 --verbose

# 2. Run on full dataset
python qwen_ui_generator.py \
  --input-dir ./input_annotations \
  --output-dir ./output_annotations \
  --model-path /path/to/model \
  --batch-size 8 \
  --num-gpus 4 \
  --verbose

# 3. Monitor progress
tail -f ./logs/qwen_generation.log

# 4. Check GPU usage
watch -n 1 nvidia-smi
```

## Output Structure

```
output_annotations/
├── file1.json  # Original data + functionality_variations + purpose
├── file2.json
└── ...

logs/
└── qwen_generation.log  # Detailed processing logs
```

## License

Internal use for Samsung Y26 Project

## Support

For issues or questions, check the logs in `./logs/qwen_generation.log` for detailed error messages.
