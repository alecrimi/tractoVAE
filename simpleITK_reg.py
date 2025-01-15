import SimpleITK as sitk
from pathlib import Path

# Define input/output directories
input_dir = "in_plane"
output_file = "registered_stack.tif"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

# Load all TIFF files into a list
tiff_files = sorted([str(f) for f in Path(input_dir).glob("*.tif")])
images = [sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32) for f in tiff_files]  # Ensure float32 at loading

# Use the first image as the fixed reference image
fixed_image = images[0]

# Initialize registration parameters
registration = sitk.ImageRegistrationMethod()
#registration.SetMetricAsMeanSquares()  #
#registration.SetMetricAsCorrelation()  
registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)
# Simpler metric for similar images
registration.SetOptimizerAsGradientDescent(learningRate=0.01, numberOfIterations=1000)
registration.SetInterpolator(sitk.sitkLinear)
#set lr=1 iter more than 500, and correlation

# Use a rigid 2D transformation (no need for moving image here)
initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    fixed_image,  # Use the fixed image as both parameters initially
    sitk.Euler2DTransform(),
    sitk.CenteredTransformInitializerFilter.MOMENTS
)
registration.SetInitialTransform(initial_transform, inPlace=False)

# Multi-resolution strategy
registration.SetShrinkFactorsPerLevel([8, 4, 2, 1])
registration.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# List to store aligned images
aligned_images = [fixed_image]  # Start with the fixed image

# Process each image in the stack (starting from the second image)
for i in range(1, len(images)):
#for i in range(1, 5):
    print(f"Registering image {i+1}/{len(images)}...")

    # Moving image
    moving_image = images[i]

    # Perform registration
    transform = registration.Execute(fixed_image, moving_image)
    aligned_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Append the aligned image to the list
    aligned_images.append(aligned_image)

# Combine all aligned images into a multi-page TIFF
print("Saving the registered stack as a single TIFF file...")
multi_page_image = sitk.JoinSeries(aligned_images)

# Define a flag to control whether scaling is applied
apply_scaling = True  # Set to False if you don't want to scale

if apply_scaling:
    print("Scaling the registered stack to 10% of its original size...")
    # Scale down the multi-page image to 10% of its original size
    scale_factor = 0.1  # 10% of the original size
    original_size = multi_page_image.GetSize()
    original_spacing = multi_page_image.GetSpacing()

    # Calculate the new size and spacing
    new_size = [int(sz * scale_factor) for sz in original_size[:2]] + [original_size[2]]  # Only scale width and height, not depth
    new_spacing = [sp / scale_factor for sp in original_spacing[:2]] + [original_spacing[2]]  # Adjust spacing for width and height

    # Apply scaling with Resample
    scaled_image = sitk.Resample(
        multi_page_image,
        new_size,
        sitk.Transform(),  # Identity transform
        sitk.sitkLinear,  # Interpolation method
        multi_page_image.GetOrigin(),
        new_spacing,
        multi_page_image.GetDirection(),
        0.0,  # Default pixel value for areas outside the original image
        multi_page_image.GetPixelID()
    )

    # Save the scaled-down image
    scaled_output_file = "scaled_" + output_file
    sitk.WriteImage(scaled_image, scaled_output_file, useCompression=True)
    print(f"Scaled registered image stack saved to {scaled_output_file}.")
else:
    print("Skipping scaling as apply_scaling is set to False.")

# Save the original registered stack without scaling
sitk.WriteImage(multi_page_image, output_file, useCompression=True)
print(f"Registered image stack saved to {output_file}.")
