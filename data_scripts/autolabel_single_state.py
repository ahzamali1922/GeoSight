import os
import rasterio
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import csv

# --- Configuration ---
STATE_NAME = 'Delhi'
INPUT_FOLDER = r'E:\labeled tiles\Delhi_tiles_8band_overlap_50'
OUTPUT_MASK_DIR = r'C:\Users\Mohvijay-sch\Desktop\GeoSight2\Delhi_Masks'
LOG_FILE = os.path.join(OUTPUT_MASK_DIR, 'labeling_log.csv')

# Thresholds for this specific state
THRESH = {'ndvi': 0.25, 'ndbi': 0.05, 'ndwi': 0.10}

def create_pseudo_label(filename):
    in_path = os.path.join(INPUT_FOLDER, filename)
    
    # Naming format: state_name_tile_number_mask.tif
    # Example: tile_100.tif -> Delhi_tile_100_mask.tif
    new_mask_name = f"{STATE_NAME}_{filename.replace('.tif', '_mask.tif')}"
    out_path = os.path.join(OUTPUT_MASK_DIR, new_mask_name)
    
    if os.path.exists(out_path):
        return (True, new_mask_name, 'already_exists')

    try:
        with rasterio.open(in_path) as src:
            green = src.read(2).astype(np.float32)
            red   = src.read(3).astype(np.float32)
            nir   = src.read(4).astype(np.float32)
            swir  = src.read(6).astype(np.float32)

            nan_frac = np.isnan(nir).mean()
            if nan_frac > 0.10:
                return (False, new_mask_name, f'NaN>{nan_frac:.2%}')

            eps = 1e-7
            green = np.nan_to_num(green); red  = np.nan_to_num(red)
            nir   = np.nan_to_num(nir);   swir = np.nan_to_num(swir)

            ndvi = (nir - red)   / (nir + red   + eps)
            ndwi = (green - nir) / (green + nir  + eps)
            ndbi = (swir - nir)  / (swir + nir   + eps)

            mask = np.zeros(ndvi.shape, dtype=np.uint8)
            mask[ndvi > THRESH['ndvi']] = 1
            mask[(ndbi > THRESH['ndbi']) & (ndvi < THRESH['ndvi'])] = 2
            mask[ndwi > THRESH['ndwi']] = 3

            profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1, nodata=255)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        return (True, new_mask_name, 'ok')

    except Exception as e:
        return (False, new_mask_name, str(e))

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Could not find folder {INPUT_FOLDER}")
        exit(1)
        
    tasks = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.tif')]
    print(f"Generating pseudo-labels for {STATE_NAME} ({len(tasks)} images)...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(create_pseudo_label, tasks), total=len(tasks)))

    success = [r for r in results if r[0]]
    failed  = [r for r in results if not r[0]]

    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'status', 'note'])
        for ok, name, note in results:
            writer.writerow([name, 'ok' if ok else 'fail', note])

    print(f"Done: {len(success)} succeeded, {len(failed)} failed.")
    print(f"Log saved to {LOG_FILE}")