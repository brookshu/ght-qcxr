'''
helper file for looking at DICOM metadata and pixel data during development of 
rotation classifier. Not used in main GHT-qCXR pipeline.
'''

import pydicom
from PIL import Image
import numpy as np
import os

# directory = '../frommartin/rotated'
# directory = '../frommartin/underexposed'
# directory = '../frommartin/optimal'
directory = 'rotation_detector/training/cxr'

# for loop to iterate through all DICOM files in a directory
for filename in os.listdir(directory):
    if filename.endswith('.dcm'):
        dcm_path = os.path.join(directory, filename)
        #print(f"cxr/{filename}")
        dcm = pydicom.dcmread(dcm_path)
        try:
            pixel_array = dcm.pixel_array.astype(float)
        except Exception as e:
            #print(f"Error reading pixel array for {filename}: {e}")
            print(f"{filename}")
            continue

        # Auto-Windowing
        if pixel_array.max() != pixel_array.min():
            scaled = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        else:
            scaled = pixel_array * 0    

        image = Image.fromarray(np.uint8(scaled))
        # if hasattr(dcm, "PhotometricInterpretation") and dcm.PhotometricInterpretation == "MONOCHROME1":
        #     image = ImageOps.invert(image)
        # image.show()  # or save with image.save('output.png')
