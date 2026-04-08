# Program 4: Vector Quantization

## Aim
To implement Vector Quantization (VQ) by mapping input vectors to their nearest codebook vectors.

## Theory
Vector Quantization is a lossy compression and clustering-based technique where input vectors are represented by the nearest vector from a finite set called a codebook.

Key ideas:
- A codebook contains representative vectors (centroids/code vectors).
- For each input vector, compute distance to each code vector.
- Assign the input vector to the nearest code vector.

Typically Euclidean distance is used:

$$d(x,c_i)=\|x-c_i\|_2$$

The output is a quantized dataset where each original vector is replaced by a codebook vector, reducing representation complexity.

## Description
1. Define sample 2D input vectors.
2. Define initial codebook vectors (centroids).
3. For each input vector, compute Euclidean distance to all codebook vectors.
4. Find the nearest codebook index using minimum distance.
5. Replace each input vector with its nearest codebook vector.
6. Print original data, codebook, and quantized data.

## Code
```python
import numpy as np

# Sample input vectors
data = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10]])

# Initial codebook (centroids)
codebook = np.array([[2, 3], [9, 9]])

# Function to find nearest code vector
def vector_quantization(data, codebook):
    quantized = []

    for vector in data:
        distances = np.linalg.norm(codebook - vector, axis=1)
        index = np.argmin(distances)
        quantized.append(codebook[index])

    return np.array(quantized)

# Perform vector quantization
quantized_vectors = vector_quantization(data, codebook)

print("Original Data:\n", data)
print("Codebook:\n", codebook)
print("Quantized Data:\n", quantized_vectors)
```

## Expected Output
```text
Original Data:
 [[ 1  2]
 [ 2  3]
 [ 3  4]
 [ 8  9]
 [ 9 10]]
Codebook:
 [[2 3]
 [9 9]]
Quantized Data:
 [[2 3]
 [2 3]
 [2 3]
 [9 9]
 [9 9]]
```
