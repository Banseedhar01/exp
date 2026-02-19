import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from Config import Config


class ActionDataset:
    """Base dataset class for handling CSV data and preprocessing"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.preprocessed_data = None
        self._read_dataset()
    
    def _read_dataset(self):
        try:
            self.data = pd.read_csv(self.dataset_path)
        except Exception as e:
            print(f"Error reading dataset: {e}")
            self.data = pd.DataFrame()
    
    def _scale_coordinates(self, left, top, right, bottom, orig_width, orig_height):
        # Scale coordinates to 1000 bins [0, 999]
        scaled_left = max(0, min(1000, round(left * 1000 / orig_width)))
        scaled_top = max(0, min(1000, round(top * 1000 / orig_height)))
        scaled_right = max(0, min(1000, round(right * 1000 / orig_width)))
        scaled_bottom = max(0, min(1000, round(bottom * 1000 / orig_height)))
        
        return scaled_left, scaled_top, scaled_right, scaled_bottom
    
    def _format_task_input(self, prefix):
        if prefix.startswith("UI_ACTION"):
            command = prefix.replace("UI_ACTION", "").strip()
            return f"<UI_ACTION>{command}"
        return prefix
    
    def load_image(self, image_id):
        try:
            image_path = f"{Config.image_dir}/{image_id}.png"
            image = Image.open(image_path).convert('RGB')
            if Config.RESIZE_IMAGES:
                image = image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color='white')
    
    def preprocess(self):
        if self.data.empty:
            print("No data to preprocess")
            return
        
        self.preprocessed_data = []
        skipped_count = 0
        
        for idx, row in self.data.iterrows():
            image_id = str(row['image_id']) if not pd.isna(row['image_id']) else ''
            input_text = str(row['input'])
            name = str(row['name'])
            
            left = int(row['left'])
            top = int(row['top'])
            right = int(row['right'])
            bottom = int(row['bottom'])
            
            orig_width = int(row['width'])
            orig_height = int(row['height'])
            
            if not image_id:
                skipped_count += 1
                continue
            
            prefix = f"UI_ACTION {input_text}".strip()
            
            if Config.RESIZE_IMAGES:
                scaled_left, scaled_top, scaled_right, scaled_bottom = self._scale_coordinates(
                    left, top, right, bottom, orig_width, orig_height
                )
                suffix = f"{name} <loc_{int(scaled_left)}> <loc_{int(scaled_top)}> <loc_{int(scaled_right)}> <loc_{int(scaled_bottom)}>"
            else:
                suffix = f"{name} <loc_{int(left)}> <loc_{int(top)}> <loc_{int(right)}> <loc_{int(bottom)}>"
            
            self.preprocessed_data.append({
                "image_id": image_id.strip(),
                "prefix": prefix.strip(),
                "suffix": suffix.strip()
            })
        
        print(f"DEBUG preprocess(): Total rows in CSV: {len(self.data)}")
        print(f"DEBUG preprocess(): Rows skipped (no image_id): {skipped_count}")
        print(f"DEBUG preprocess(): Valid samples: {len(self.preprocessed_data)}")
    
    def getData(self):
        if self.preprocessed_data is None:
            self.preprocess()
        
        if Config.TEST_MODE:
            return self.preprocessed_data[:Config.TEST_SAMPLE_SIZE]
        
        return self.preprocessed_data
    
    def getItem(self, index=0):
        if self.preprocessed_data is None:
            self.preprocess()
        
        if not self.preprocessed_data or index >= len(self.preprocessed_data):
            return None
            
        return self.preprocessed_data[index]


class FlorenceActionDataset(Dataset):
    """PyTorch Dataset for Florence-2 training with lazy image loading"""
    
    def __init__(self, action_dataset):
        self.action_dataset = action_dataset
        print(f"DEBUG FlorenceActionDataset.__init__: Loading data...")
        self.data = action_dataset.getData()
        print(f"DEBUG FlorenceActionDataset.__init__: Dataset size = {len(self.data)}")
        print(f"DEBUG FlorenceActionDataset.__init__: Data loaded successfully")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['prefix'], item['suffix'], item['image_id']