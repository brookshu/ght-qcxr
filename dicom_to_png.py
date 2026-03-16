import os
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

# ===== SETTINGS =====
input_folder = "rotation_detector/training/cxr/"
output_folder = "all_png_output"
os.makedirs(output_folder, exist_ok=True)

def dicom_to_numpy(ds):
    """
    Convert DICOM pixel data into normalized numpy array.
    Handles:
    - Rescale slope/intercept
    - MONOCHROME1 inversion
    - normalization
    """

    img = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # Normalize intensities
    img_min = np.min(img)
    img_max = np.max(img)

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    # Handle MONOCHROME1 (invert grayscale)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = 1.0 - img

    # Convert to 8-bit
    img = (img * 255).astype(np.uint8)

    return img

def convert_folder():
    files = [f for f in os.listdir(input_folder) if not f.startswith('.')]

    for f in tqdm(files):
        in_path = os.path.join(input_folder, f)

        try:
            ds = pydicom.dcmread(in_path)
            img = dicom_to_numpy(ds)

            out_name = os.path.splitext(f)[0] + ".png"
            out_path = os.path.join(output_folder, out_name)

            Image.fromarray(img).save(out_path)

        except Exception as e:
            print(f"Skipping {f}: {e}")

if __name__ == "__main__":
    convert_folder()