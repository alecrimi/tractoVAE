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

# Step 3: Create synthetic diffusion data with precise unit vectors
# First vector is [0,0,0] for b0, others are unit vectors
bvecs = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [-1., 0., 0.],
    [0., 1., 0.],
    [0., -1., 0.],
    [0., 0., 1.],
    [0., 0., -1.],
    [1., 1., 0.],
    [-1., -1., 0.],
    [1., 0., 1.],
    [-1., 0., -1.],
    [0., 1., 1.],
    [0., -1., -1.]
], dtype=np.float64)

# Precisely normalize non-zero vectors
for i in range(1, len(bvecs)):  # Skip first vector (b0)
    norm = np.sqrt(np.sum(bvecs[i] ** 2))
    if norm > 0:
        bvecs[i] = bvecs[i] / norm

# Verify that vectors are unit length (except b0)
for i in range(1, len(bvecs)):
    norm = np.sqrt(np.sum(bvecs[i] ** 2))
    if not np.isclose(norm, 1.0, rtol=1e-7):
        raise ValueError(f"Vector {i} is not unit length: {norm}")

# Create corresponding b-values
bvals = np.zeros(len(bvecs), dtype=np.float64)
bvals[1:] = 1000.0  # Set non-b0 values to 1000

# Create the gradient table
gtab = gradient_table(bvals, bvecs, b0_threshold=50)

# Create synthetic diffusion weighted images
data = np.zeros(image_stack.shape + (len(bvals),))
for z in range(image_stack.shape[0]):
    for y in range(image_stack.shape[1]):
        for x in range(image_stack.shape[2]):
            # Create diffusion tensor from eigenvalues
            D = np.diag(eigvals[z, y, x])
            # Calculate signal for each gradient direction
            for i in range(len(bvals)):
                if i == 0:  # b0 image
                    data[z, y, x, i] = image_stack[z, y, x]
                else:
                    # S = S0 * exp(-b * g^T * D * g)
                    g = bvecs[i]
                    S = image_stack[z, y, x] * np.exp(-bvals[i] * g.dot(D).dot(g))
                    data[z, y, x, i] = S

# Fit the tensor model
model = TensorModel(gtab)
fit = model.fit(data)

# Get FA
FA = fit.fa

# Create stopping criterion
stopping_criterion = BinaryStoppingCriterion(FA > 0.2)

# Generate peaks
sphere = default_sphere
peaks = peaks_from_model(
    model=model,
    data=data[..., 0],
    sphere=sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25
)

# Generate streamlines
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
