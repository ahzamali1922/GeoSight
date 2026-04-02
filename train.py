import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from tqdm import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
import random

# Suppress annoying rasterio warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
# Prevent GDAL/rasterio from locking up threads on Windows
os.environ['GDAL_NUM_THREADS'] = '1'

# --- Configuration ---
IMG_DIR = r"C:\Users\Mohvijay-sch\Desktop\GeoSight2\GeoSight_Consolidated_Dataset\Images"
# Fixed mask directory pointing to your new SSD folder
MASK_DIR = r"C:\Users\Mohvijay-sch\Desktop\GeoSight2\Consolidated_Masked"

BATCH_SIZE = 16  # Adjusted for RTX A6000; 48 can be too aggressive if images are large
EPOCHS = 30
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Definition ---
class GeoSightDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Use provided filenames or read them if none are provided
        if filenames is None:
            self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        else:
            self.filenames = filenames
            
        print(f"? Initialized Dataset split with {len(self.filenames)} images.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Mask name: "Delhi_tile_1.tif" -> "Delhi_tile_1_mask.tif"
        mask_name = img_name.replace(".tif", "_mask.tif")
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            # 1. Read Image (Bands 1-6 are active)
            with rasterio.open(img_path) as src:
                image = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)
            
            # Basic normalization (Scale to 0-1)
            image = np.clip(image / 10000.0, 0, 1)
            
            # Add this safety net to catch NaNs in the raw data
            if np.isnan(image).any() or np.isinf(image).any():
                image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
                
            image = np.transpose(image, (1, 2, 0)) # CHW to HWC for Albumentations
    
            # 2. Read Mask
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int64) 
                
            # CRITICAL FIX for CUDA "index out of bounds" crash:
            mask[mask > 3] = 0
            mask[mask < 0] = 0
            mask = mask.astype(np.uint8)
    
            # 3. Apply Transforms
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask'].long() # CrossEntropy needs Long
                
            return image, mask
            
        except Exception as e:
            # DO NOT delete files in the Dataset/DataLoader. Just skip and retry logging.
            print(f"\n?? Error reading {img_name}: {e}. Skipping and picking random replacement.")
            new_idx = random.randint(0, len(self.filenames) - 1)
            return self.__getitem__(new_idx)

# --- Data Augmentation ---
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
    ToTensorV2(),
])

val_transform = A.Compose([
    ToTensorV2(),
])

# --- Main Script ---
def main():
    print(f"?? Initializing GeoSight Training on GPU: {torch.cuda.get_device_name(0)}")
        
    # Get all files and split the filenames FIRST
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.tif')]
    random.seed(42)
    random.shuffle(all_files)
    
    train_size = int(0.9 * len(all_files))
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    
    # Initialize Datasets strictly separate so validation data isn't leaked with augmentations
    train_dataset = GeoSightDataset(IMG_DIR, MASK_DIR, filenames=train_files, transform=train_transform)
    # Validation needs ToTensorV2 to format the data correctly for PyTorch (no spatial augmentations)
    val_dataset = GeoSightDataset(IMG_DIR, MASK_DIR, filenames=val_files, transform=val_transform) 
    
    # DataLoader with A6000 Performance Tweaks
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, # Lowered to 4 to prevent GDAL memory locks
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Model: U-Net++ with EfficientNet-B4 Backbone
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet", 
        in_channels=6, 
        classes=4, # Background, Rural, Urban, Water
    ).to(device)
    
    # Auto-Resume from crash if checkpoint exists
    recovery_path = "geosight_recovery_checkpoint.pt"
    if os.path.exists(recovery_path):
        print(f"\n?? Found recovery checkpoint! Resuming weights from {recovery_path}...")
        model.load_state_dict(torch.load(recovery_path, map_location=device, weights_only=True))
    
    # Loss Strategy: Hybrid Dice + Focal Loss with eps parameter appended
    dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, eps=1e-7)
    focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Enable GradScaler only if CUDA is available for AMP acceleration
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        loop = tqdm(train_loader, desc=f"Training")
        for step, (images, masks) in enumerate(loop):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True) 

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with Mixed Precision using stable bfloat16
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = dice_loss(outputs, masks) + focal_loss(outputs, masks)
                
                scaler.scale(loss).backward()
                
                # Unscale before clipping gradients
                scaler.unscale_(optimizer)
                # Clip gradients to prevent explosion (nan loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU Fallback
                outputs = model(images)
                loss = dice_loss(outputs, masks) + focal_loss(outputs, masks)
                loss.backward()
                
                # Clip gradients to prevent explosion (nan loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
            # Auto-save every 400 batches (~10% of the epoch) to prevent progress loss
            if (step + 1) % 400 == 0:
                torch.save(model.state_dict(), "geosight_recovery_checkpoint.pt")

        # Evaluation mode (Validation)
        model.eval()
        val_loss = 0.0
        # For Tracking the Accuracy and IoU (Intersection over Union)
        total_tp, total_fp, total_fn, total_tn = [], [], [], []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(images)
                        loss = dice_loss(outputs, masks) + focal_loss(outputs, masks)
                else:
                    outputs = model(images)
                    loss = dice_loss(outputs, masks) + focal_loss(outputs, masks)
                
                val_loss += loss.item()

                # Calculate metrics
                predictions = torch.argmax(outputs, dim=1).unsqueeze(1) # Get highest class score and expand
                
                # Expand dims to fit metric API
                masks = masks.unsqueeze(1).long()
                
                tp, fp, fn, tn = smp.metrics.get_stats(predictions, masks, mode='multiclass', num_classes=4)
                # Ensure memory leak doesn't happen by moving them to CPU
                total_tp.append(tp.cpu())
                total_fp.append(fp.cpu())
                total_fn.append(fn.cpu())
                total_tn.append(tn.cpu())

        # Compute average metrics across the validation set
        total_tp = torch.cat(total_tp)
        total_fp = torch.cat(total_fp)
        total_fn = torch.cat(total_fn)
        total_tn = torch.cat(total_tn)

        # Micro IoU and Accuracy
        iou_score = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="micro")
        accuracy = smp.metrics.accuracy(total_tp, total_fp, total_fn, total_tn, reduction="macro")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"? Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}")
        print(f"    Validation Accuracy: {accuracy:.4f}  |  Validation mIoU: {iou_score:.4f}")
        
        # Save Checkpoint to the specified output folder based on request although let's just make sure it stays in current dir if fine. You'd mentioned models to "come in geosight2", which is the cwd.
        torch.save(model.state_dict(), f"geosight_final_epoch_{epoch+1}.pt")

    print("?? Training Finished! All 5 states integrated into one model.")

if __name__ == '__main__':
    main()



