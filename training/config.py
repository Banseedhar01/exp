"""
Config.py — Central configuration for Florence-2 training.

All values here are defaults.  The training script (train.py) reads CLI
arguments and overwrites the attributes on this class *before* any dataset
or model code reads them, so every module that does `from Config import Config`
will automatically see the CLI-provided values.
"""


class Config:
    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_path: str = "./Florence-2-base"
    FREEZE_VISION_ENCODER: bool = False

    # Custom task tokens added to the tokenizer/embeddings via add_custom_tokens()
    CUSTOM_TASK_TOKENS: list = [
        "<UI_ACTION>",
        "<OD>",
        "<VQA>",
        "<UI_PURPOSE>",
        "<UI_EXPECTATIONS>",
        "<UI_CAPTION>",
    ]

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    EPOCHS: int = 5
    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 1e-5
    GRADIENT_ACCUMULATION_STEPS: int = 4
    MAX_GRAD_NORM: float = 1.0

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    RESIZE_IMAGES: bool = True
    IMAGE_SIZE: int = 768

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    model_output_dir: str = "./checkpoints/final"
    log_dir: str = "./logs"

    # ------------------------------------------------------------------
    # Existing datasets (CSV-based)
    # ------------------------------------------------------------------
    # ActionDataset
    commands_path: str = "./data/action.csv"
    image_dir: str = "./data/images"          # shared image dir for action + info

    # InfoDataset
    info_path: str = "./data/info.csv"

    # ------------------------------------------------------------------
    # AMEX datasets (JSON-based)
    # ------------------------------------------------------------------
    amex_od_json: str = "./data/amex_od.json"
    amex_od_image_dir: str = "./data/amex_od_images"

    amex_ui_action_json: list = ["./data/amex_ui_action.json"]   # supports multiple JSON files
    amex_ui_action_image_dir: str = "./data/amex_ui_action_images"

    vqa_json: str = "./data/vqa.json"
    vqa_image_dir: str = "./data/vqa_images"

    amex_purpose_json: list = ["./data/amex_purpose.json"]       # supports multiple JSON files
    amex_purpose_image_dir: str = "./data/amex_purpose_images"

    amex_expectation_json: str = "./data/amex_expectation.json"
    amex_expectation_image_dir: str = "./data/amex_expectation_images"

    # ------------------------------------------------------------------
    # Dataset selection flags
    # ------------------------------------------------------------------
    USE_ACTION: bool = False
    USE_INFO: bool = False
    USE_AMEX_OD: bool = False
    USE_AMEX_UI_ACTION: bool = False
    USE_VQA: bool = False
    USE_AMEX_PURPOSE: bool = False
    USE_AMEX_EXPECTATION: bool = False

    # ------------------------------------------------------------------
    # Per-dataset sample caps  (None = no limit)
    # ------------------------------------------------------------------
    MAX_ACTION: int = None
    MAX_INFO: int = None
    MAX_AMEX_OD: int = None
    MAX_AMEX_UI_ACTION: int = None
    MAX_VQA: int = None
    MAX_AMEX_PURPOSE: int = None
    MAX_AMEX_EXPECTATION: int = None

    # ------------------------------------------------------------------
    # Validation split
    # ------------------------------------------------------------------
    VAL_SPLIT: float = 0.1          # fraction of combined data held out for validation

    # ------------------------------------------------------------------
    # Test / debug mode
    # ------------------------------------------------------------------
    TEST_MODE: bool = False
    TEST_SAMPLE_SIZE: int = 64