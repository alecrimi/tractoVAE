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

# Step 2: Compute the eigenvalues and eigenvectors of the structure tensor for each voxel
eigvals = np.zeros(image_stack.shape + (3,))
eigvecs = np.zeros(image_stack.shape + (3, 3))
for z in range(image_stack.shape[0]):
    for y in range(image_stack.shape[1]):
        for x in range(image_stack.shape[2]):
            tensor = np.array([
                [Axx[z, y, x], Axy[z, y, x], Axz[z, y, x]],
                [Axy[z, y, x], Ayy[z, y, x], Ayz[z, y, x]],
                [Axz[z, y, x], Ayz[z, y, x], Azz[z, y, x]],
            ])
            eigenvals, eigenvectors = np.linalg.eigh(tensor)
            eigvals[z, y, x] = eigenvals
            eigvecs[z, y, x] = eigenvectors

# Step 3: Create minimal set of gradient directions
bvecs = np.array([
    [0., 0., 0.],  # b0
    [1., 0., 0.],
    [-1., 0., 0.],
    [0., 1., 0.],
    [0., -1., 0.],
    [0., 0., 1.],
    [0., 0., -1.],
], dtype=np.float32)  # Use float32 instead of float64 to save memory

# Normalize non-zero vectors
for i in range(1, len(bvecs)):
    bvecs[i] = bvecs[i] / np.linalg.norm(bvecs[i])

# Create b-values
bvals = np.zeros(len(bvecs), dtype=np.float32)  # Use float32
bvals[1:] = 1000.0

# Create gradient table
gtab = gradient_table(bvals, bvecs, b0_threshold=50)

# Process data in chunks to save memory
chunk_size = 50  # Adjust this based on your available memory
n_chunks = (image_stack.shape[0] + chunk_size - 1) // chunk_size

data_shape = image_stack.shape + (len(bvals),)
data = np.memmap('temp_dwi.npy', dtype=np.float32, mode='w+', shape=data_shape)

for chunk in range(n_chunks):
    start_idx = chunk * chunk_size
    end_idx = min((chunk + 1) * chunk_size, image_stack.shape[0])
    
    print(f"Processing chunk {chunk + 1}/{n_chunks}")
    
    for z in range(start_idx, end_idx):
        for y in range(image_stack.shape[1]):
            for x in range(image_stack.shape[2]):
                S0 = max(float(image_stack[z, y, x]), 1)  # Avoid zero values
                D = np.diag(np.abs(eigvals[z, y, x]))  # Ensure positive eigenvalues
                
                # Store b0 image
                data[z, y, x, 0] = S0
                
                # Calculate signal for each non-b0 direction
                for i in range(1, len(bvals)):
                    g = bvecs[i]
                    S = S0 * np.exp(-bvals[i] * g.dot(D).dot(g))
                    data[z, y, x, i] = S

# Fit the tensor model
print("Fitting tensor model...")
model = TensorModel(gtab)
fit = model.fit(data)

# Get FA map
FA = fit.fa

# Clean up temporary file
import os
os.unlink('temp_dwi.npy')

# Create stopping criterion
stopping_criterion = BinaryStoppingCriterion(FA > 0.2)

# Generate peaks
print("Generating peaks...")
peaks = peaks_from_model(
    model=model,
    data=data,
    sphere=default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    parallel=False  # Disable parallel processing to save memory
)

# Generate streamlines
print("Generating streamlines...")
seed_mask = FA > 0.2
seeds = np.argwhere(seed_mask)
streamlines_generator = LocalTracking(peaks, stopping_criterion, seeds, np.eye(4), step_size=0.5)
streamlines = Streamlines(streamlines_generator)

# Save the streamlines
header = {
    'voxel_sizes': (1.0, 1.0, 1.0),
    'voxel_order': 'LPS',
    'dim': image_stack.shape[:3],
}
save_trk("microscopy_tractography.trk", streamlines, np.eye(4), image_stack.shape[:3], header=header)
print("Streamlines saved to 'microscopy_tractography.trk'")
