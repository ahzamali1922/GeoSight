import os
import shutil
import glob
from tqdm import tqdm

def consolidate_masks():
    print("Starting consolidation of masks to SSD...")
    
    # Find all the generated mask folders inside Masked_Dataset
    MASK_FOLDERS = glob.glob(r'C:\Users\Mohvijay-sch\Desktop\GeoSight2\Masked_Dataset\*_Masks')
    
    # The new master folder for masks on your SSD (C: drive)
    OUTPUT_MASK_DIR = r'C:\Users\Mohvijay-sch\Desktop\GeoSight2\Consolidated_Masked'
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    total_moved = 0
    
    for folder_path in MASK_FOLDERS:
        state_name = os.path.basename(folder_path).split('_')[0]
        files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        
        print(f"Moving {len(files)} masks for {state_name}...")
        
        for filename in tqdm(files):
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(OUTPUT_MASK_DIR, filename)
            
            try:
                # We use move here to avoid keeping duplicates on your C: drive
                shutil.move(old_path, new_path) 
                total_moved += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")
                
    print(f"Finished! {total_moved} masks successfully moved to: {OUTPUT_MASK_DIR}")

if __name__ == '__main__':
    consolidate_masks()