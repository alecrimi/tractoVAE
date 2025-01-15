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
image_stack = io.imread('scaled_registered_stack_bins200_rate0.1_i500.tif')

# Step 1: Compute the structure tensor of the image stack
Axx, Axy, Axz, Ayy, Ayz, Azz = structure_tensor(image_stack, sigma=1)

# Step 2: Compute the eigenvalues of the structure tensor for each voxel
eigvals = np.zeros(image_stack.shape + (3,))
for z in range(image_stack.shape[0]):
    for y in range(image_stack.shape[1]):
        for x in range(image_stack.shape[2]):
            tensor = np.array([
                [Axx[z, y, x], Axy[z, y, x], Axz[z, y, x]],
                [Axy[z, y, x], Ayy[z, y, x], Ayz[z, y, x]],
                [Axz[z, y, x], Ayz[z, y, x], Azz[z, y, x]],
            ])
            eigvals[z, y, x] = np.linalg.eigh(tensor)[0]

# Step 3: Create a diffusion tensor from the eigenvalues
# Reshape the tensor data to match expected dimensions (x, y, z, 6)
# We'll use the six unique elements of the symmetric 3x3 tensor
tensor_data = np.zeros(image_stack.shape + (6,))
tensor_data[..., 0] = eigvals[..., 2]  # Dxx
tensor_data[..., 1] = eigvals[..., 1]  # Dyy
tensor_data[..., 2] = eigvals[..., 0]  # Dzz
tensor_data[..., 3] = np.zeros_like(eigvals[..., 0])  # Dxy
tensor_data[..., 4] = np.zeros_like(eigvals[..., 0])  # Dxz
tensor_data[..., 5] = np.zeros_like(eigvals[..., 0])  # Dyz

# Step 4: Create gradient table and fit tensor model
bvals = np.array([0, 1000, 1000])
bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
gtab = gradient_table(bvals, bvecs)

# Fit the tensor model
model = TensorModel(gtab)
fit = model.fit(tensor_data)

# Get FA and ensure it's 3D
FA = fit.fa
if FA.ndim > 3:
    FA = np.squeeze(FA)  # Remove any singleton dimensions

# Step 5: Create stopping criterion
stopping_criterion = BinaryStoppingCriterion(FA > 0.2)

# Step 6: Generate peaks
sphere = default_sphere
peaks = peaks_from_model(model, image_stack, sphere, relative_peak_threshold=0.5)

# Step 7: Generate streamlines
seed_mask = FA > 0.2
seeds = np.argwhere(seed_mask)
streamlines_generator = LocalTracking(peaks, stopping_criterion, seeds, np.eye(4), step_size=0.5)
streamlines = Streamlines(streamlines_generator)

# Step 8: Save the streamlines
header = {
    'voxel_sizes': (1.0, 1.0, 1.0),
    'voxel_order': 'LPS',
    'dim': image_stack.shape[:3],
}
save_trk("microscopy_tractography.trk", streamlines, np.eye(4), image_stack.shape[:3], header=header)
print("Streamlines saved to 'microscopy_tractography.trk'")
