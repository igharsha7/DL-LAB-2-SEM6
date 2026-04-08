# Program 5: Self-Organization Feature Maps (SOM)

## Aim
To implement a Self-Organizing Feature Map (SOM) that mimics the behavior of a small class of biological neural networks by organizing input patterns through competitive learning.

## Theory
A Self-Organizing Map (SOM), proposed by Kohonen, is an unsupervised neural network that maps high-dimensional input data onto a lower-dimensional representation while preserving neighborhood relationships.

Core concepts:
- Each neuron has a weight vector of the same dimension as input data.
- For each input vector, compute distance to all neurons.
- The neuron with minimum distance is the Best Matching Unit (BMU).
- Update BMU weights toward the input vector.

Typical update rule:

$$w_{new} = w_{old} + \eta (x - w_{old})$$

Where:
- $w$ is neuron weight vector.
- $x$ is input vector.
- $\eta$ is learning rate.

Through repeated training epochs, neurons self-organize to represent input clusters/features.

## Description
1. Define input data points.
2. Randomly initialize weight vectors for neurons.
3. Set learning rate and number of epochs.
4. For each input sample, compute Euclidean distance to neuron weights.
5. Identify the BMU (winning neuron).
6. Update only the BMU weight toward the current input.
7. Repeat for all epochs.
8. Print final trained weights representing the feature map.

## Code
```python
#Write a program to implement Self-organization Feature Maps to mimic the actions of a small class of biological neural networks
import numpy as np

# Input data
data = np.array([[0.2, 0.8],
                 [0.1, 0.9],
                 [0.8, 0.2],
                 [0.9, 0.1]])

# Initialize weight vectors for 2 neurons
weights = np.random.rand(2, 2)

learning_rate = 0.5
epochs = 10

for epoch in range(epochs):
    for x in data:
        # Compute distance between input and neurons
        distances = np.linalg.norm(weights - x, axis=1)

        # Find winning neuron (Best Matching Unit)
        winner = np.argmin(distances)

        # Update weight of winning neuron
        weights[winner] = weights[winner] + learning_rate * (x - weights[winner])

print("Final Weights (Feature Map):")
print(weights)
```

## Expected Output
```text
Final Weights (Feature Map):
[[0.84980469 0.15019531]
 [0.15019531 0.84980469]]
```

Note: Exact final weights vary each run because initial weights are random.
