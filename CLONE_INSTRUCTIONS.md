# Clone and Setup Instructions

## Quick Setup on Different PC

Follow these simple steps to get the code on a different computer:

### Step 1: Clone the Repository

```bash
git clone https://github.com/Banseedhar01/exp.git
cd exp
```

### Step 2: Checkout the Branch

```bash
git checkout qwen-ui-generator
```

### Step 3: Verify Files

```bash
ls
# You should see:
# - qwen_ui_generator.py
# - requirements.txt
# - README.md
# - run_example.sh
# - sample_input.json
# - sample_expected_output.json
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run Test

```bash
python qwen_ui_generator.py \
  --input-dir /path/to/your/input/jsons \
  --output-dir /path/to/your/output/jsons \
  --model-path /group-volume/SRIB-Bixby-Screen-AI/k.anup/Qwen-models/Qwen3-30B-A3B-Instruct-2507 \
  --test --test-samples 5 --verbose
```

---

## Alternative: Clone Specific Branch Directly

If you only want the specific branch:

```bash
git clone -b qwen-ui-generator https://github.com/Banseedhar01/exp.git
cd exp
```

---

## Troubleshooting

**If branch doesn't exist locally:**
```bash
git fetch origin
git checkout qwen-ui-generator
```

**To see all available branches:**
```bash
git branch -a
```

**To pull latest changes:**
```bash
git pull origin qwen-ui-generator
```

---

## Summary

The simplest way:
```bash
# One-liner to clone and switch to branch
git clone https://github.com/Banseedhar01/exp.git && cd exp && git checkout qwen-ui-generator
```

Then install dependencies and run!
