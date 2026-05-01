Basic learnings about Numpy - 


## What is NumPy?

NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides:
- Powerful N-dimensional array objects
- Fast mathematical operations on arrays
- Tools for working with large datasets
- Foundation for other libraries (Pandas, Matplotlib, Scikit-learn)

## Why NumPy?

### NumPy vs Python Lists

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Speed | Slower | **50x faster** |
| Memory | More memory | Less memory |
| Functionality | Basic | Rich mathematical functions |
| Type | Mixed types | Single type (homogeneous) |
| Size | Dynamic | Fixed size (can reshape) |

**Example - Speed Comparison:**
```python
import numpy as np
import time

# Python list
python_list = list(range(1000000))
start = time.time()
python_result = [x * 2 for x in python_list]
print(f"Python list: {time.time() - start:.4f} seconds")

# NumPy array
numpy_array = np.arange(1000000)
start = time.time()
numpy_result = numpy_array * 2
print(f"NumPy array: {time.time() - start:.4f} seconds")
```

## Installation

```bash
# Install NumPy
pip install numpy

# Check version
pip show numpy
```

## Importing NumPy

```python
# Standard convention
import numpy as np

# Check version
print(np.__version__)
```

## Creating Arrays

### 1. From Python Lists

```python
# 1D array
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)  # [1 2 3 4 5]

# 2D array (matrix)
arr2d = np.array([[1, 2, 3], 
                  [4, 5, 6]])
print(arr2d)
# [[1 2 3]
#  [4 5 6]]

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]]])
```

### 2. Using Built-in Functions

```python
# Array of zeros
zeros = np.zeros(5)              # [0. 0. 0. 0. 0.]
zeros_2d = np.zeros((3, 4))      # 3x4 matrix of zeros

# Array of ones
ones = np.ones(5)                # [1. 1. 1. 1. 1.]
ones_2d = np.ones((2, 3))        # 2x3 matrix of ones

# Array with a constant value
full = np.full(5, 7)             # [7 7 7 7 7]
full_2d = np.full((2, 3), 3.14)  # 2x3 matrix of 3.14

# Range of values
arange = np.arange(10)           # [0 1 2 3 4 5 6 7 8 9]
arange_step = np.arange(0, 10, 2)  # [0 2 4 6 8]

# Evenly spaced values
linspace = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]

# Identity matrix
identity = np.eye(3)             # 3x3 identity matrix

# Random arrays
random = np.random.random(5)           # 5 random values [0, 1)
random_int = np.random.randint(0, 10, 5)  # 5 random integers [0, 10)
random_normal = np.random.randn(5)     # 5 random from normal distribution
```

## Array Attributes

```python
arr = np.array([[1, 2, 3, 4], 
                [5, 6, 7, 8]])

# Shape (dimensions)
print(arr.shape)      # (2, 4) - 2 rows, 4 columns

# Number of dimensions
print(arr.ndim)       # 2

# Total number of elements
print(arr.size)       # 8

# Data type
print(arr.dtype)      # int64 (or int32 on Windows)

# Size of each element (in bytes)
print(arr.itemsize)   # 8 bytes

# Total memory used
print(arr.nbytes)     # 64 bytes (8 elements × 8 bytes)
```

## Data Types

```python
# Integer
int_arr = np.array([1, 2, 3], dtype=np.int32)

# Float
float_arr = np.array([1.0, 2.5, 3.7], dtype=np.float64)

# Boolean
bool_arr = np.array([True, False, True], dtype=bool)

# String
str_arr = np.array(['a', 'b', 'c'], dtype=str)

# Change data type (casting)
arr = np.array([1, 2, 3])
arr_float = arr.astype(np.float64)
print(arr_float)  # [1. 2. 3.]
```

## Basic Array Information

```python
arr = np.array([[1, 2, 3], 
                [4, 5, 6]])

# Print array info
print("Array:")
print(arr)
print(f"Shape: {arr.shape}")
print(f"Dimensions: {arr.ndim}")
print(f"Size: {arr.size}")
print(f"Data type: {arr.dtype}")
```

**Output:**
```
Array:
[[1 2 3]
 [4 5 6]]
Shape: (2, 3)
Dimensions: 2
Size: 6
Data type: int64
```

## Reshaping Arrays

```python
# Original 1D array
arr = np.arange(12)  # [0 1 2 3 4 5 6 7 8 9 10 11]

# Reshape to 2D
arr_2d = arr.reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Reshape to 3D
arr_3d = arr.reshape(2, 2, 3)

# Flatten back to 1D
flattened = arr_2d.flatten()  # or arr_2d.ravel()

# Automatic dimension calculation
auto = arr.reshape(3, -1)  # -1 means "figure it out" → (3, 4)
```

## Array Copying

```python
# View (reference to original)
arr = np.array([1, 2, 3])
view = arr.view()
view[0] = 100
print(arr)    # [100   2   3] - Original changed!

# Copy (independent copy)
arr = np.array([1, 2, 3])
copy = arr.copy()
copy[0] = 100
print(arr)    # [1 2 3] - Original unchanged
```

## Common Mistakes to Avoid

❌ **Wrong:**
```python
# Trying to create array with mixed types
arr = np.array([1, 2, 'three'])  # All become strings!
print(arr.dtype)  # <U21 (string)
```

✅ **Right:**
```python
# Keep consistent types
arr = np.array([1, 2, 3])
print(arr.dtype)  # int64
```

---

❌ **Wrong:**
```python
# Modifying a view thinking it's a copy
arr = np.array([1, 2, 3])
arr2 = arr
arr2[0] = 100
print(arr)  # [100   2   3] - Oops!
```

✅ **Right:**
```python
# Use .copy() when you need independence
arr = np.array([1, 2, 3])
arr2 = arr.copy()
arr2[0] = 100
print(arr)  # [1 2 3] - Safe!
```

## Practice Exercises

Try these yourself:

1. Create a 1D array with numbers from 0 to 9
2. Create a 3x3 matrix of all zeros
3. Create a 2x5 matrix filled with the number 7
4. Create an array with values from 10 to 50 with step 5
5. Create a 4x4 identity matrix
6. Create an array of 10 random integers between 1 and 100
7. Reshape a 1D array [0,1,2,3,4,5,6,7,8,9,10,11] into a 3x4 matrix
8. Check the shape, dtype, and size of your created arrays

## Key Takeaways

✅ NumPy is much faster than Python lists for numerical operations
✅ Arrays must have the same data type (homogeneous)
✅ Use `np.array()` to create from lists, or built-in functions for special arrays
✅ Important attributes: `.shape`, `.ndim`, `.size`, `.dtype`
✅ Use `.copy()` when you need an independent copy
✅ Reshaping doesn't change the data, just its organization
