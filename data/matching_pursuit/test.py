import numpy as np
import cupy as cp

# Create a NumPy array
np_array = np.array([1, 2, 3, 4, 5])

# Create a CuPy array
cp_array = cp.array([1, 2, 3, 4, 5])

# Check if the array is a NumPy array
if np_array.__class__ == np.ndarray:
    print("np_array is a NumPy array")

# Check if the array is a CuPy array
if cp_array.__class__ == cp.ndarray:
    print("cp_array is a CuPy array", type(cp_array))
