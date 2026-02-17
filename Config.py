"""
Configuration for Florence-2 Training
Supports multiple dataset formats: OD (Object Detection) and FUN (Function/Action)
"""

class Config:
    # Dataset paths
    commands_path = "../datasets/guiact_dataset/web_single_click_dataset_with_bbox.csv"
    caption_path = "data/captions.json"
    expectation_path = "data/expectations.json"
    image_dir = "../datasets/guiact_dataset/images/"
    model_output_dir = "./florence2_finetuned"
    
    # Dataset format configuration
    # Options: "OD" (Object Detection), "FUN" (Function/Action), "AUTO" (auto-detect)
    DATASET_FORMAT = "AUTO"
    
    # For new dataset formats, specify the JSON file path or directory path
    # Set to None to use the legacy CSV format (commands_path)
    DATASET_JSON_PATH = None  # e.g., "data/od_dataset.json" or "data/fun_data/" (directory)
    
    # Task tokens for model vocabulary
    # COMMAND token is used for function/action tasks
    # OD token is used for object detection tasks
    TOKENS = ["UI_ACTION", "CAPTION", "EXPECTATION", "OD", "COMMAND"]
    CUSTOM_TASK_TOKENS = ["<UI_ACTION>", "<CAPTION>", "<EXPECTATION>", "<OD>", "<COMMAND>"]
    
    # Image processing
    IMAGE_SIZE = 768
    RESIZE_IMAGES = True
    
    # Training hyperparameters
    LEARNING_RATE = 1e-5
    EPOCHS = 7
    BATCH_SIZE = 2
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1.0
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Training optimizations
    MIXED_PRECISION = True
    TEST_MODE = False
    TEST_SAMPLE_SIZE = 5
    FREEZE_VISION_ENCODER = False
    
    # Legacy task configuration (for backward compatibility)
    LOC_TOKEN_FORMAT = "<loc_{}>"
    TASK_ACTION = "ACTION"
    TASK_CAPTION = "CAPTION"
    TASK_EXPECTATION = "EXPECTATION"
    TASK_LOCATE = "OD"
    
    @classmethod
    def use_new_dataset_format(cls) -> bool:
        """Check if using new JSON dataset format."""
        return cls.DATASET_JSON_PATH is not None
    
    @classmethod
    def get_dataset_path(cls) -> str:
        """Get the appropriate dataset path based on configuration."""
        if cls.use_new_dataset_format():
            return cls.DATASET_JSON_PATH
        return cls.commands_path
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings."""
        import os
        
        # Check dataset path exists
        dataset_path = cls.get_dataset_path()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Check image directory exists
        if not os.path.exists(cls.image_dir):
            raise FileNotFoundError(f"Image directory not found: {cls.image_dir}")
        
        # Validate format type
        valid_formats = ["OD", "FUN", "AUTO"]
        if cls.DATASET_FORMAT not in valid_formats:
            raise ValueError(f"DATASET_FORMAT must be one of {valid_formats}, got: {cls.DATASET_FORMAT}")
        
        # Validate hyperparameters
        if cls.BATCH_SIZE < 1:
            raise ValueError(f"BATCH_SIZE must be >= 1, got: {cls.BATCH_SIZE}")
        
        if cls.EPOCHS < 1:
            raise ValueError(f"EPOCHS must be >= 1, got: {cls.EPOCHS}")
        
        if cls.LEARNING_RATE <= 0:
            raise ValueError(f"LEARNING_RATE must be > 0, got: {cls.LEARNING_RATE}")
        
        return True


config = Config()
__all__ = ['Config', 'config']
