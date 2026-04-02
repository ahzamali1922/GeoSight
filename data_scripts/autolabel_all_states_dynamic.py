import os
import rasterio
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import csv

# --- Configuration ---
STATES = [
    ('Kanpur', r'E:\labeled tiles\Kanpur_tiles_8band_overlap_50 (1)\Kanpur_tiles_8band_overlap_50')
]

BASE_OUT_DIR = r'C:\Users\Mohvijay-sch\Desktop\GeoSight2'
os.makedirs(BASE_OUT_DIR, exist_ok=True)

def profile_state(state_name, input_folder, n=200):
    print(f"\n[{state_name}] Profiling to find optimal thresholds...")
    files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    sample = random.sample(files, min(n, len(files)))
    
    ndvi_all, ndwi_all, ndbi_all = [], [], []
    
    for f in tqdm(sample, desc=f"Reading {state_name} samples"):
        try:
            with rasterio.open(os.path.join(input_folder, f)) as src:
                g   = np.nan_to_num(src.read(2).astype(np.float32))
                r   = np.nan_to_num(src.read(3).astype(np.float32))
                nir = np.nan_to_num(src.read(4).astype(np.float32))
                sw  = np.nan_to_num(src.read(6).astype(np.float32))
                eps = 1e-7
                ndvi_all.append((nir - r)   / (nir + r   + eps))
                ndwi_all.append((g - nir)   / (g   + nir + eps))
                ndbi_all.append((sw  - nir) / (sw  + nir + eps))
        except:
            continue

    if len(ndvi_all) == 0:
        print(f"Warning: Could not read any tiles for {state_name}. Using default thresholds.")
        return {'ndvi': 0.3, 'ndwi': 0.15, 'ndbi': 0.05}

    ndvi_flat = np.concatenate([x.ravel() for x in ndvi_all])
    ndwi_flat = np.concatenate([x.ravel() for x in ndwi_all])
    ndbi_flat = np.concatenate([x.ravel() for x in ndbi_all])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{state_name} — Index Distributions ({len(sample)} tile sample)', fontsize=12)

    for ax, data, name, color in zip(
        axes,
        [ndvi_flat, ndwi_flat, ndbi_flat],
        ['NDVI', 'NDWI', 'NDBI'],
        ['green', 'blue', 'orange']
    ):
        # Sub-sample for faster plotting if data size is huge
        subset = np.random.choice(data, min(len(data), 1000000), replace=False)
        ax.hist(subset, bins=120, color=color, alpha=0.7, edgecolor='none')
        p25, p50, p75 = np.percentile(subset, [25, 50, 75])
        ax.axvline(p25, color='red',    linestyle='--', linewidth=1, label=f'p25={p25:.3f}')
        ax.axvline(p50, color='black',  linestyle='-',  linewidth=1, label=f'p50={p50:.3f}')
        ax.axvline(p75, color='purple', linestyle='--', linewidth=1, label=f'p75={p75:.3f}')
        ax.set_title(name)
        ax.set_xlabel('Index value')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(BASE_OUT_DIR, f'{state_name}_index_profile.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    t_ndvi = np.percentile(ndvi_flat, 75)
    t_ndwi = np.percentile(ndwi_flat, 75)
    t_ndbi = np.percentile(ndbi_flat, 50)
    
    print(f"--- Recommended & Applied thresholds for {state_name} ---")
    print(f"  NDVI (Rural)  → {t_ndvi:.3f}")
    print(f"  NDWI (Water)  → {t_ndwi:.3f}")
    print(f"  NDBI (Urban)  → {t_ndbi:.3f}")
    
    return {'ndvi': t_ndvi, 'ndwi': t_ndwi, 'ndbi': t_ndbi}


def process_tile(args):
    # Unpack arguments
    filename, state_name, input_folder, output_mask_dir, thresh = args
    in_path = os.path.join(input_folder, filename)
    
    new_mask_name = f"{state_name}_{filename.replace('.tif', '_mask.tif')}"
    out_path = os.path.join(output_mask_dir, new_mask_name)
    
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
            # 1: Vegetation, 2: Built-up, 3: Water, 0: Background
            mask[ndvi > thresh['ndvi']] = 1
            mask[(ndbi > thresh['ndbi']) & (ndvi < thresh['ndvi'])] = 2
            mask[ndwi > thresh['ndwi']] = 3

            profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1, nodata=255)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        return (True, new_mask_name, 'ok')

    except Exception as e:
        return (False, new_mask_name, str(e))


if __name__ == "__main__":
    print("Starting dynamic thresholding and masking for all states...")

    tasks_total = 0
    success_total = 0
    failed_total = 0

    for state_name, input_folder in STATES:
        if not os.path.exists(input_folder):
            print(f"Skipping {state_name}, folder not found at: {input_folder}\n")
            continue
            
        # 1. Dynamically calculate thresholds for the current state
        thresh = profile_state(state_name, input_folder, n=200)
        
        # 2. Setup output directory specific to the state
        output_mask_dir = os.path.join(BASE_OUT_DIR, f"{state_name}_Masks")
        os.makedirs(output_mask_dir, exist_ok=True)
        log_file = os.path.join(output_mask_dir, 'labeling_log.csv')
        
        # 3. Create the parallel processing tasks
        files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
        state_tasks = [(f, state_name, input_folder, output_mask_dir, thresh) for f in files]
        tasks_total += len(state_tasks)
        
        print(f"Generating {len(state_tasks)} masks for {state_name} into folder: {output_mask_dir} ...")
        
        # 4. Execute multiprocessor
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(process_tile, state_tasks), total=len(state_tasks), desc=f"{state_name} Labels"))
            
        # 5. Log results for this state
        success = [r for r in results if r[0]]
        failed  = [r for r in results if not r[0]]
        
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'status', 'note'])
            for ok, name, note in results:
                writer.writerow([name, 'ok' if ok else 'fail', note])
                
        print(f"Finished {state_name}: {len(success)} succeeded, {len(failed)} failed.\n")
        print("-" * 50)
        
        success_total += len(success)
        failed_total += len(failed)

    print("\n==============================")
    print(f"GRAND TOTAL PROCESS COMPLETED.")
    print(f"Total Successfully created masks: {success_total}")
    print(f"Total Failed tiles: {failed_total}")
    print("All thresholds and profile graph plots have been saved inside the main directory.")
    print("==============================")