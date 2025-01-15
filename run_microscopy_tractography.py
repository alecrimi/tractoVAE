import numpy as np
from skimage.feature import structure_tensor
from skimage import io
from dipy.data import default_sphere
from dipy.reconst.dti import TensorModel, gradient_table
from dipy.tracking.streamline import Streamlines
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.direction import peaks_from_model
from dipy.io.streamline import save_trk

# Load your microscopy image stack (3D volume)
# Assuming the stack is loaded as a numpy array (Z x Y x X)
image_stack = io.imread('scaled_registered_stack_bins200_rate0.1_i500.tif')

# Step 1: Compute the structure tensor of the image stack
# Using a 3D Gaussian sigma for a stack (change sigma depending on image resolution)
Axx, Axy, Axz, Ayy, Ayz, Azz = structure_tensor(image_stack, sigma=1)

# Step 2: Compute the eigenvalues of the structure tensor for each voxel
eigvals = np.zeros(image_stack.shape + (3,))  # Store 3 eigenvalues per voxel

for z in range(image_stack.shape[0]):
    for y in range(image_stack.shape[1]):
        for x in range(image_stack.shape[2]):
            # Construct the 3x3 structure tensor at voxel (z, y, x)
            tensor = np.array([
                [Axx[z, y, x], Axy[z, y, x], Axz[z, y, x]],
                [Axy[z, y, x], Ayy[z, y, x], Ayz[z, y, x]],
                [Axz[z, y, x], Ayz[z, y, x], Azz[z, y, x]],
            ])
            # Compute eigenvalues of the tensor and store them
            eigvals[z, y, x] = np.linalg.eigh(tensor)[0]  # Sorted eigenvalues

# Step 3: Create a diffusion tensor from the eigenvalues
tensor_data = np.zeros(image_stack.shape + (3, 3))
tensor_data[..., 0, 0] = eigvals[..., 2]  # Lambda 1 (largest eigenvalue)
tensor_data[..., 1, 1] = eigvals[..., 1]  # Lambda 2 (middle eigenvalue)
tensor_data[..., 2, 2] = eigvals[..., 0]  # Lambda 3 (smallest eigenvalue)

# Step 4: Fit the tensor model using Dipy's TensorModel
# Create a fake gradient table (assuming two gradient directions for simplicity)
bvals = [0, 1000]  # Replace with actual b-values if available
bvecs = np.array([[1, 0, 0], [0, 1, 0]])  # Replace with actual b-vectors if available
gtab = gradient_table(bvals=bvals, bvecs=bvecs)
model = TensorModel(gtab)
fit = model.fit(tensor_data)

# Step 5: Create a stopping criterion for tractography (using FA > threshold)
FA = fit.fa  # Fractional Anisotropy map
stopping_criterion = BinaryStoppingCriterion(FA > 0.2)

# Step 6: Use the peak directions for local tracking
sphere = default_sphere  # Define directions used for tracking
peaks = peaks_from_model(model, image_stack, sphere, relative_peak_threshold=0.5)

# Step 7: Generate streamlines (seed points based on FA threshold)
seed_mask = FA > 0.2
seeds = np.argwhere(seed_mask)  # Generate seed points in voxel space

# Perform local tracking
streamlines_generator = LocalTracking(peaks, stopping_criterion, seeds, np.eye(4), step_size=0.5)
streamlines = Streamlines(streamlines_generator)

# Step 8: Save the streamlines to a .trk file for TrackVis
header = {
    'voxel_sizes': (1.0, 1.0, 1.0),  # Assume isotropic voxels (adjust if necessary)
    'voxel_order': 'LPS',            # Left-Posterior-Superior, adjust as needed
    'dim': image_stack.shape[:3],    # Dimensions of the stack
}
save_trk("microscopy_tractography.trk", streamlines, np.eye(4), image_stack.shape[:3], header=header)

print("Streamlines saved to 'microscopy_tractography.trk'")
