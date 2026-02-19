import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from Config import Config


class InfoDataset:
    """Dataset class for processing transformed dataset with CAPTION and EXPECTATIONS tasks"""
    
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
    
    def load_image(self, image_id):
        """Load image from image directory"""
        try:
            image_path = f"{Config.image_dir}/{image_id}.png"
            image = Image.open(image_path).convert('RGB')
            if Config.RESIZE_IMAGES:
                image = image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    
    def preprocess(self):
        """Preprocess the dataset into (prefix, suffix, image_id) format"""
        if self.data.empty:
            print("No data to preprocess")
            return
        
        self.preprocessed_data = []
        skipped_count = 0
        
        for idx, row in self.data.iterrows():
            # Extract fields from dataset
            image_id = str(row['image_id'])
            input_text = str(row['input'])
            output_text = str(row['output'])
            
            top = int(row['top'])
            left = int(row['left'])
            right = int(row['right'])
            bottom = int(row['bottom'])
            
            width = int(row['width'])
            height = int(row['height'])
            
            # Skip if image_id is empty
            if not image_id:
                skipped_count += 1
                continue
            
            # Rescale coordinates to 1000 bins [0, 999]
            # x1, x2 are scaled by width; y1, y2 are scaled by height
            scaled_x1 = max(0, min(1000, round(left * 1000 / width)))
            scaled_y1 = max(0, min(1000, round(top * 1000 / height)))
            scaled_x2 = max(0, min(1000, round(right * 1000 / width)))
            scaled_y2 = max(0, min(1000, round(bottom * 1000 / height)))
            
            # Format: input + <loc_x1> <loc_y1> <loc_x2> <loc_y2>
            prefix = f"{input_text} <loc_{int(scaled_x1)}> <loc_{int(scaled_y1)}> <loc_{int(scaled_x2)}> <loc_{int(scaled_y2)}>"
            
            # Suffix is the output text
            suffix = output_text.strip()
            
            self.preprocessed_data.append({
                "image_id": image_id.strip(),
                "prefix": prefix.strip(),
                "suffix": suffix
            })
        
        print(f"DEBUG InfoDataset preprocess(): Total rows in CSV: {len(self.data)}")
        print(f"DEBUG InfoDataset preprocess(): Rows skipped (no image_id): {skipped_count}")
        print(f"DEBUG InfoDataset preprocess(): Valid samples: {len(self.preprocessed_data)}")
    
    def getData(self):
        """Get preprocessed data"""
        if self.preprocessed_data is None:
            self.preprocess()
        
        if Config.TEST_MODE:
            return self.preprocessed_data[:Config.TEST_SAMPLE_SIZE]
        
        return self.preprocessed_data
    
    def getItem(self, index=0):
        """Get a single item from preprocessed data"""
        if self.preprocessed_data is None:
            self.preprocess()
        
        if not self.preprocessed_data or index >= len(self.preprocessed_data):
            return None
            
        return self.preprocessed_data[index]


class FlorenceInfoDataset(Dataset):
    """PyTorch Dataset for InfoDataset with lazy image loading"""
    
    def __init__(self, info_dataset):
        self.info_dataset = info_dataset
        print(f"DEBUG FlorenceInfoDataset.__init__: Loading data...")
        self.data = info_dataset.getData()
        print(f"DEBUG FlorenceInfoDataset.__init__: Dataset size = {len(self.data)}")
        print(f"DEBUG FlorenceInfoDataset.__init__: Data loaded successfully")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['prefix'], item['suffix'], item['image_id']