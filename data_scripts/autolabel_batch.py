import os
import rasterio
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime

# I have updated the input folders to match the exact paths we found in your E:\labeled tiles folder!
INPUT_FOLDERS = [
    r'E:\labeled tiles\Delhi_tiles_8band_overlap_50',
    r'E:\labeled tiles\Haryana_tiles_labeled',
    r'E:\labeled tiles\Kanpur_tiles_8band_overlap_50 (1)',
    r'E:\labeled tiles\Sikkim_tiles_labeled',
    r'E:\labeled tiles\UK_tiles_labeled'
]
OUTPUT_MASK_DIR = r'C:\Users\Mohvijay-sch\Desktop\GeoSight2'
LOG_FILE = r'C:\Users\Mohvijay-sch\Desktop\GeoSight2\labeling_log.csv'
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# --- Per-region thresholds (Phase 2 improvement baked in) ---
REGION_THRESHOLDS = {
    'Delhi':   {'ndvi': 0.25, 'ndbi': 0.05, 'ndwi': 0.10},
    'Haryana': {'ndvi': 0.35, 'ndbi': 0.05, 'ndwi': 0.15},
    'Kanpur':  {'ndvi': 0.28, 'ndbi': 0.08, 'ndwi': 0.12},
    'Sikkim':  {'ndvi': 0.45, 'ndbi': 0.02, 'ndwi': 0.20},
    'UK':      {'ndvi': 0.40, 'ndbi': 0.03, 'ndwi': 0.18},
}

def get_region(folder_path):
    for region in ['Delhi', 'Haryana', 'Kanpur', 'Sikkim', 'UK']:
        if region.lower() in folder_path.lower():
            return region
    return 'Haryana'  # default fallback

def create_pseudo_label(file_info):
    folder, filename = file_info
    region = get_region(folder)
    in_path = os.path.join(folder, filename)
    # Define a custom output name based on region and filename to avoid overwrites
    new_mask_name = f"{region}_{filename.replace('.tif', '_mask.tif')}"
    out_path = os.path.join(OUTPUT_MASK_DIR, new_mask_name)
    
    # If mask already exists, skip it to save time if script is paused/resumed
    if os.path.exists(out_path):
        return (True, new_mask_name, 'already_exists')

    thresh = REGION_THRESHOLDS[region]

    try:
        with rasterio.open(in_path) as src:
            # Read active bands: B2=Green, B3=Red, B4=NIR, B6=SWIR2
            green = src.read(2).astype(np.float32)
            red   = src.read(3).astype(np.float32)
            nir   = src.read(4).astype(np.float32)
            swir  = src.read(6).astype(np.float32)

            # --- Quality check: skip mostly-NaN tiles ---
            nan_frac = np.isnan(nir).mean()
            if nan_frac > 0.10:
                return (False, filename, f'NaN>{nan_frac:.2%}')

            eps = 1e-7
            # Replace NaN with 0 for index calculation
            green = np.nan_to_num(green); red  = np.nan_to_num(red)
            nir   = np.nan_to_num(nir);   swir = np.nan_to_num(swir)

            ndvi = (nir - red)   / (nir + red   + eps)
            ndwi = (green - nir) / (green + nir  + eps)
            ndbi = (swir - nir)  / (swir + nir   + eps)

            # --- Classification (water overrides all) ---
            mask = np.zeros(ndvi.shape, dtype=np.uint8)
            mask[ndvi > thresh['ndvi']] = 1                            # Rural/Vegetation
            mask[(ndbi > thresh['ndbi']) & (ndvi < thresh['ndvi'])] = 2 # Urban/Roads
            mask[ndwi > thresh['ndwi']] = 3                            # Water

            # --- Class balance check (flag degenerate tiles) ---
            total = mask.size
            class_counts = np.bincount(mask.ravel(), minlength=4)
            dominant_frac = class_counts.max() / total

            profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1, nodata=255)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)
                
            if dominant_frac > 0.98:
                # Nearly-single-class tile — save but flag it
                label = f'dominant_class_{class_counts.argmax()}'
                return (True, new_mask_name, label)

        return (True, new_mask_name, 'ok')

    except Exception as e:
        return (False, new_mask_name, str(e))


if __name__ == "__main__":
    tasks = []
    print("Finding all .tif files...")
    for fld in INPUT_FOLDERS:
        if os.path.exists(fld):
            tasks.extend([(fld, f) for f in os.listdir(fld) if f.endswith('.tif')])
        else:
            print(f"Warning: Folder not found -> {fld}")

    print(f"Generating pseudo-labels for {len(tasks)} images...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(create_pseudo_label, tasks), total=len(tasks)))

    success = [r for r in results if r[0]]
    failed  = [r for r in results if not r[0]]

    # Save log
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'status', 'note'])
        for ok, name, note in results:
            writer.writerow([name, 'ok' if ok else 'fail', note])

    print(f"Done: {len(success)} succeeded, {len(failed)} failed.")
    print(f"Log saved to {LOG_FILE}")
