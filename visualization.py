import os
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

# --- CONFIG ---
IMG_DIR = r"C:\Users\Mohvijay-sch\Desktop\GeoSight2\GeoSight_Consolidated_Dataset\Images"
MODEL_PATH = r"checkpoints\final_weight_epoch.pt"
OUTPUT_DIR = r"E:\GeoSight_Final_Predictions"
BATCH_SIZE = 128  # High batch size for A6000 inference
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)
        image = np.clip(image / 10000.0, 0, 1)
        return torch.from_numpy(image), img_name

def run_full_inference():
    # Load Model
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=6, classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    dataset = InferenceDataset(IMG_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    print(f"🚀 Starting Full Inference on {len(dataset)} images...")

    with torch.no_grad():
        for images, names in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)

            # Save each mask in the batch
            for i in range(preds.shape[0]):
                out_path = os.path.join(OUTPUT_DIR, names[i].replace(".tif", "_pred.tif"))
                # Note: We save as a simple 1-channel TIF
                with rasterio.open(
                    out_path, 'w', driver='GTiff',
                    height=256, width=256, count=1,
                    dtype='uint8'
                ) as dst:
                    dst.write(preds[i], 1)

    print(f"✅ All predictions saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_full_inference()