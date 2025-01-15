#Stack together the tiff files and register them

from PIL import Image
import os

import numpy as np
from skimage import io, transform
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import tifffile

def stack_together(directory):
 # List of TIFF files to stack
 tiff_files = [f for f in os.listdir(directory) if f.endswith('.tiff') #tiff_files = ["image1.tiff", "image2.tiff", "image3.tiff"]
 output_file = "stacked_output.tiff"
 
 # Open the first image
 images = [Image.open(f) for f in tiff_files]
 first_image = images[0]
 
 # Save as multi-page TIFF
 first_image.save(output_file, save_all=True, append_images=images[1:])


# Now register everything to the first slice


def register_images_to_first_slice(tiff_stack):
    # Extract the first slice as the reference
    reference_slice = tiff_stack[0]

    registered_stack = []
    registered_stack.append(reference_slice)  # First slice is already aligned

    for slice in tiff_stack[1:]:
        # Register the current slice to the first slice
        shift, _, _ = register_translation(reference_slice, slice)
        aligned_slice = fourier_shift(np.fft.fftn(slice), shift)
        registered_stack.append(np.fft.ifftn(aligned_slice).real)

    return np.array(registered_stack)

def load_tiff_stack(directory):
    # Load all TIFF images from the directory and return as a list
    tiff_files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
    tiff_files.sort()  # Sort files if necessary
    tiff_stack = []

    for tiff_file in tiff_files:
        img = io.imread(os.path.join(directory, tiff_file))
        tiff_stack.append(img)

    return np.array(tiff_stack)

def save_registered_stack(registered_stack, output_path):
    # Save the registered stack as a multi-page TIFF file
    tifffile.imwrite(output_path, registered_stack)

def main():
    # Directory containing the TIFF stack
    directory = '/path/to/tiff_stack_folder'

    stack_together(directory):
    
    # Load the TIFF stack
    tiff_stack = load_tiff_stack(directory)
    
    # Register all slices to the first slice
    registered_stack = register_images_to_first_slice(tiff_stack)
    
    # Output path for the registered stack
    output_path = '/path/to/output/registered_stack.tiff'
    
    # Save the registered stack
    save_registered_stack(registered_stack, output_path)

    print(f"Registered stack saved to {output_path}")

if __name__ == '__main__':
    main()
