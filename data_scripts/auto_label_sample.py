import numpy as np
import rasterio
import matplotlib.pyplot as plt

# The input image path
img_path = r'E:\labeled tiles\UK_tiles_labeled\tile_316.tif'
out_path = r'C:\Users\Mohvijay-sch\Desktop\GeoSight2\sample_mask.tif'
out_png  = r'C:\Users\Mohvijay-sch\Desktop\GeoSight2\sample_mask.png'

print(f"Processing: {img_path}")

with rasterio.open(img_path) as src:
    # Read the necessary bands
    green = src.read(2)  # B3
    nir = src.read(4)    # B8
    ndvi = src.read(7)   # NDVI
    ndbi = src.read(8)   # NDBI
    
    profile = src.profile

# Calculate NDWI (Normalized Difference Water Index) = (Green - NIR) / (Green + NIR)
# Adding a small epsilon to avoid division by zero
ndwi = (green - nir) / (green + nir + 1e-8)

# Initialize an empty mask with zeros (Background = 0)
mask = np.zeros(ndvi.shape, dtype=np.uint8)

# Rule 1: Water (Class 3)
# Usually NDWI > 0.1 indicates water
water_threshold = 0.1
mask[ndwi > water_threshold] = 3

# Rule 2: Vegetation (Class 1)
# Usually NDVI > 0.25 indicates healthy vegetation
veg_threshold = 0.25
mask[(ndvi > veg_threshold) & (mask == 0)] = 1

# Rule 3: Built-up / Urban (Class 2)
# Usually NDBI > 0.0 indicates built-up land
built_threshold = 0.0
mask[(ndbi > built_threshold) & (mask == 0)] = 2

# Check how many pixels belong to each class
unique, counts = np.unique(mask, return_counts=True)
class_names = {0: "Background/Barren", 1: "Vegetation", 2: "Built-up", 3: "Water"}

print("\n--- Auto-Labeling Results ---")
for val, count in zip(unique, counts):
    print(f"Class {val} ({class_names[val]}): {count} pixels")

# Save as a single-band TIF file
profile.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=None
)

with rasterio.open(out_path, 'w', **profile) as dst:
    dst.write(mask, 1)

print(f"\nSaved mask successfully to: {out_path}")

# Optional: Save a visual PNG version for you to easily look at
plt.imshow(mask, cmap='viridis', vmin=0, vmax=3)
plt.colorbar(ticks=[0, 1, 2, 3], format=plt.FuncFormatter(lambda val, loc: class_names.get(int(val), "")))
plt.title("Auto-Generated Label Mask")
plt.savefig(out_png)
print(f"Saved visual PNG successfully to: {out_png}")
