import numpy as np
import os
import scipy.io as sio

output_base_path = "data/Darcy-merged/merge_{}.npy"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

# Load raw training data from .mat files
for j in range(1, 6):
    print(f"Processing file {j}...")
    training_raw_data = sio.loadmat(f"data/training/darcy/darcy_{j}.mat")
    a = training_raw_data["thresh_a_data"]
    u = training_raw_data["thresh_p_data"]

    # Iterate over each index i
    for i in range(10000):
        # Load the files
        f1 = a[i]
        f2 = u[i]

        # Transform the arrays, normalize to (-1, 1) for diffusion model
        # NOTE: Scales are different for different data types. See the generation codes for details.
        # NOTE: The transformation is the inverse of the transformation in the generation codes.
        f1_transformed = f1 * 0.2 - 1.5
        f2_transformed = f2 * 115 - 0.9

        # Combine them into a new array with a shape [H, W, 2]
        combined = np.stack((f1_transformed, f2_transformed), axis=-1)

        # Save the combined array to a new .npy file
        output_file_path = output_base_path.format(i+(j-1)*10000)
        np.save(output_file_path, combined)

        assert combined.shape == (128, 128, 2)
        if i % 500 == 0:
            print(f"Saved combined array for index {i} to {output_file_path}")
            print("Min:", combined.min(), "Max:", combined.max())

print("Finished processing all files.")