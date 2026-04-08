# Program 3: Principal Components Extraction

## Aim
To implement Principal Component Analysis (PCA) for extracting a smaller set of variables that represents a multivariate dataset.

## Theory
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms correlated variables into a new set of uncorrelated variables called principal components.

Key points:
- The first principal component captures maximum variance in the data.
- Each subsequent component captures the next highest variance under orthogonality constraints.
- PCA is computed using eigenvalues and eigenvectors of the covariance matrix.

For a centered data matrix $X$, PCA steps are:
1. Compute covariance matrix.
2. Find eigenvalues and eigenvectors.
3. Sort eigenvectors in decreasing order of eigenvalues.
4. Select top $k$ eigenvectors.
5. Project data to get reduced representation.

## Description
1. Define a sample multivariate dataset.
2. Perform mean normalization (centering).
3. Compute covariance matrix of centered data.
4. Compute eigenvalues and eigenvectors.
5. Sort components by descending explained variance.
6. Select top $k$ principal components.
7. Transform original data into lower-dimensional space.
8. Print original shape, reduced shape, principal components, and transformed data.

## Code
```python
#Write a program to implement Principal Components extraction to represent a multivariate data table as smaller set of variables
import numpy as np

# Sample multivariate dataset (rows = observations, columns = variables)
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2.0, 1.6],
              [1.0, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# Step 1: Mean normalization (center the data)
mean_vec = np.mean(X, axis=0)
X_centered = X - mean_vec

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_centered.T)

# Step 3: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvectors by descending eigenvalues
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Step 5: Select top k principal components (k = 1 or 2 etc.)
k = 1
principal_components = sorted_eigenvectors[:, :k]

# Step 6: Transform the original data
X_reduced = np.dot(X_centered, principal_components)

print("Original Data Shape:", X.shape)
print("Reduced Data Shape:", X_reduced.shape)
print("Principal Components:\n", principal_components)
print("Transformed Data:\n", X_reduced)
```

## Expected Output
```text
Original Data Shape: (10, 2)
Reduced Data Shape: (10, 1)
Principal Components:
 [[ 0.6778734 ]
 [ 0.73517866]]
Transformed Data:
 [[ 0.82797019]
 [-1.77758033]
 [ 0.99219749]
 [ 0.27421042]
 [ 1.67580142]
 [ 0.9129491 ]
 [-0.09910944]
 [-1.14457216]
 [-0.43804614]
 [-1.22382056]]
```

Note: The sign of principal components may appear flipped in some runs/environments (equally valid in PCA), but reduced representation remains equivalent.
