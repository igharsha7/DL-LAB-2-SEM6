# Program 1: Generalized Brain State in a Box (BSB) Neural Network

## Aim
To design and implement a generalized Brain State in a Box (BSB) neural network to store bipolar patterns and recall the closest stored pattern from a noisy input.

## Theory
Brain State in a Box (BSB) is an auto-associative recurrent neural network model used for pattern storage and recall. It operates with continuous or bipolar state values and converges to stable states (attractors) corresponding to learned patterns.

Key points:
- BSB uses a recurrent weight matrix computed from training patterns.
- State updates are iterative and follow:

$$x(k+1)=f(\alpha x(k)+Wx(k))$$

Where:
- $x(k)$ is the current state vector.
- $W$ is the weight matrix.
- $\alpha$ is a scaling factor.
- $f(\cdot)$ is the box (saturation) activation function.

The box activation constrains each component to the range $[-1, 1]$, which helps stabilization:
- If value > 1, output becomes 1
- If value < -1, output becomes -1
- Else value remains unchanged

## Description
1. Define a box activation function using clipping in the range $[-1,1]$.
2. Train the BSB network by computing the weight matrix as the sum of outer products of training patterns.
3. Remove self-connections by setting diagonal elements of the weight matrix to zero.
4. Provide a noisy test pattern.
5. Iteratively update the state vector using the BSB update rule.
6. Observe convergence and print the final recalled pattern.

## Code
```python
import numpy as np

# Saturation (box) activation function
#If value > 1 → output becomes 1
#If value < -1 → output becomes -1
#If value between -1 and 1 → value remains same
def box_activation(x):
    return np.clip(x, -1, 1)

# Training: compute weight matrix
def train_bsb(patterns):
    patterns = np.array(patterns)

    #Initialize Weight Matrix
    W = np.zeros((patterns.shape[1], patterns.shape[1]))
    #Calculate Weight Matrix W=∑(p×pT)
    for p in patterns:
        p = p.reshape(-1,1)
        W += np.dot(p, p.T)

    np.fill_diagonal(W, 0)   # remove self-connections
    return W

# Recall process
#This function recalls stored pattern from noisy input.
# W-Weight matrix, x-Input vector, alpha-scaling parameter (constant that is multipled with present state vector x(k) before updating next state),
#iterations-number of updates
def recall_bsb(W, x, alpha=0.5, iterations=10):
    x = np.array(x)

    #Iterative Update
    # BSB update equation x(k+1)=f(αx(k)+Wx(k))
    # x(k)- current state, W- weight matrix, α- scaling parameter, f-box activation
    for i in range(iterations):
        x = box_activation(alpha*x + np.dot(W, x))
        #Print how the state vector evolves toward stable memory.
        print(f"Iteration {i+1}: {x}")

    return x

# Training patterns
patterns = [
    [1, -1, 1, -1],
    [-1, 1, -1, 1]
]

# Train network- computes the weight matrix
W = train_bsb(patterns)
print("Weight Matrix:\n", W)

# Test pattern (noisy input)
test = [1, -1, 1, -0.5]

# Recall stored pattern
#The network updates the vector repeatedly until it converges to the nearest stored memory.
result = recall_bsb(W, test)
#Final Output
print("Final Recalled Pattern:", result)
```

## Expected Output
```text
Weight Matrix:
 [[ 0. -2.  2. -2.]
 [-2.  0. -2.  2.]
 [ 2. -2.  0. -2.]
 [-2.  2. -2.  0.]]
Iteration 1: [ 1. -1.  1. -1.]
Iteration 2: [ 1. -1.  1. -1.]
Iteration 3: [ 1. -1.  1. -1.]
Iteration 4: [ 1. -1.  1. -1.]
Iteration 5: [ 1. -1.  1. -1.]
Iteration 6: [ 1. -1.  1. -1.]
Iteration 7: [ 1. -1.  1. -1.]
Iteration 8: [ 1. -1.  1. -1.]
Iteration 9: [ 1. -1.  1. -1.]
Iteration 10: [ 1. -1.  1. -1.]
Final Recalled Pattern: [ 1. -1.  1. -1.]
```