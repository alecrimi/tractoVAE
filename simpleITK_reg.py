import SimpleITK as sitk
from pathlib import Path

# Define input/output directories
input_dir = "in_plane"
output_file = "registered_stack.tif"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

# Load all TIFF files into a list
tiff_files = sorted([str(f) for f in Path(input_dir).glob("*.tif")])
images = [sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32) for f in tiff_files]  # Ensure float32 at loading

# Use the first image as the initial reference
reference_image = images[0]

# Initialize registration parameters
registration = sitk.ImageRegistrationMethod()
#registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration.SetMetricAsMeanSquares()
registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
registration.SetInterpolator(sitk.sitkLinear)

# Use a rigid 2D transformation
registration.SetInitialTransform(sitk.CenteredTransformInitializer(
    reference_image,
    reference_image,
    sitk.Euler2DTransform(),  # Use 2D transform for 2D images
    sitk.CenteredTransformInitializerFilter.GEOMETRY
))
registration.SetOptimizerScalesFromPhysicalShift()

# List to store aligned images
aligned_images = [reference_image]  # Start with the reference image

# Process each image in the stack (starting from the second image)
for i in range(1, len(images)):
    print(f"Registering image {i+1}/{len(images)}...")

    # Ensure the moving image is float32
    moving_image = images[i]

    # Perform registration
    transform = registration.Execute(reference_image, moving_image)
    aligned_image = sitk.Resample(moving_image, reference_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Append the aligned image to the list
    aligned_images.append(aligned_image)

    # Update the reference image to the newly aligned image
    reference_image = aligned_image

# Combine all aligned images into a multi-page TIFF
print("Saving the registered stack as a single TIFF file...")
multi_page_image = sitk.JoinSeries(aligned_images)
sitk.WriteImage(multi_page_image, output_file, useCompression=True)

print(f"Registered image stack saved to {output_file}.")
