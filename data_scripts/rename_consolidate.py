import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# These are your folders on the E: drive containing the originals
RAW_IMAGE_FOLDERS = [
    r'E:\labeled tiles\Delhi_tiles_8band_overlap_50', 
    r'E:\labeled tiles\Haryana_tiles_labeled',
    r'E:\labeled tiles\Kanpur_tiles_8band_overlap_50',
    r'E:\labeled tiles\Sikkim_tiles_labeled',
    r'E:\labeled tiles\UK_tiles_labeled'
]

# Create a clean master folder for your consolidated dataset
OUTPUT_MASTER_DIR = r'E:\GeoSight_Consolidated_Dataset\Images'
os.makedirs(OUTPUT_MASTER_DIR, exist_ok=True)

def rename_and_consolidate():
    print(f"Starting consolidation of original images...")
    
    total_processed = 0
    
    for folder_path in RAW_IMAGE_FOLDERS:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        # Determine state name from the folder path 
        # e.g., 'Delhi_tiles_8band_overlap_50' -> 'Delhi'
        raw_folder_name = os.path.basename(folder_path)
        state_name = raw_folder_name.split('_')[0]
        
        # Find all .tif files
        files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        
        print(f"Processing {len(files)} tiles for {state_name}...")
        
        # Loop with progress bar
        for filename in tqdm(files):
            old_path = os.path.join(folder_path, filename)
            
            # Prefix the state name to match mask convention 
            # e.g., 'tile_1.tif' -> 'Delhi_tile_1.tif'
            new_filename = f"{state_name}_{filename}"
            new_path = os.path.join(OUTPUT_MASTER_DIR, new_filename)
            
            try:
                # Using move to avoid duplicating 130GB of data
                shutil.move(old_path, new_path) 
                total_processed += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
                
    print(f"Finished! {total_processed} original images prefixed and consolidated into: {OUTPUT_MASTER_DIR}")

if __name__ == '__main__':
    rename_and_consolidate()
