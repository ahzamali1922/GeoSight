import rasterio
import numpy as np
import os
import random
from tqdm import tqdm

folders = [
    r'E:\Delhi_tiles_8band_overlap_50', 
    r'E:\Haryana_all_combined_tif_tiles',
    r'E:\Kanpur_tiles_8band_overlap_50',
    r'E:\Sikkim_tiles_8band_overlap_50pct_clean',
    r'E:\UK_tiles_all_tif'
]

def get_folder_stats(folder_path, num_samples=100):
    if not os.path.exists(folder_path):
        return None
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    if len(files) == 0:
        return None
    
    # Pick 100 random files (or all if less than 100)
    samples = random.sample(files, min(num_samples, len(files)))
    
    # Store sums to calculate grand means
    all_band_means = []
    
    print(f"\n--- Profiling {len(samples)} images in: {os.path.basename(folder_path)} ---")
    
    for filename in tqdm(samples):
        try:
            with rasterio.open(os.path.join(folder_path, filename)) as src:
                # Read all 8 bands at once
                data = src.read().astype('float32')
                # Use nanmean to avoid 'nan' pixels
                means = np.nanmean(data, axis=(1, 2))
                all_band_means.append(means)
        except:
            continue
            
    if all_band_means:
        final_stats = np.nanmean(all_band_means, axis=0)
        for i, val in enumerate(final_stats, 1):
            print(f"Band {i} Global Mean: {val:.4f}")
    return final_stats

# Run the profiler
for folder in folders:
    get_folder_stats(folder)