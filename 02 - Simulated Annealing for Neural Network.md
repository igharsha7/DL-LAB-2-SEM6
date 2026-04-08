# Program 2: Simulated Annealing for Neural Network

## Aim
To implement the Simulated Annealing optimization technique and obtain a near-optimal solution for minimizing an objective function.

## Theory
Simulated Annealing (SA) is a probabilistic optimization method inspired by the annealing process in metallurgy. It avoids getting stuck in local minima by occasionally accepting worse solutions with a probability controlled by temperature.

Main ideas:
- Start with an initial random solution.
- Generate a neighboring solution.
- If the neighbor is better, accept it.
- If worse, accept it with probability:

$$P = e^{-\Delta/T}$$

Where:
- $\Delta$ is the increase in cost.
- $T$ is the current temperature.

As $T$ decreases gradually, the algorithm becomes less likely to accept worse solutions and converges toward a near-optimal result.

## Description
1. Define objective function $f(x)=x^2$.
2. Initialize random solution $x$ in range $[-10,10]$.
3. Set initial temperature, minimum temperature, and cooling rate.
4. Generate a neighbor by adding a small random perturbation.
5. Compute energy difference between new and current solutions.
6. Apply acceptance rule (always accept better, probabilistically accept worse).
7. Reduce temperature repeatedly until stopping criterion is met.
8. Print final solution and its function value.

## Code
```python
#Start with a random solution
#Try a neighbor solution
#If better → accept it
#If worse → accept it with some probability
#Gradually reduce temperature (T)
#Finally reach a near-optimal solution

import random
import math

# Objective function
def f(x):
    return x**2

# Simulated Annealing
def simulated_annealing():
    x = random.uniform(-10, 10)   # initial solution
    T = 1000                      # initial temperature
    T_min = 1e-3                  # stopping temperature
    alpha = 0.9                   # cooling rate

    while T > T_min:
        x_new = x + random.uniform(-1, 1)   # neighbor solution

        # change in energy (cost)
        delta = f(x_new) - f(x)

        # acceptance condition
        if delta < 0 or random.random() < math.exp(-delta / T):
            x = x_new

        T = T * alpha   # reduce temperature

    return x

# Run the algorithm
result = simulated_annealing()
print("Optimal solution:", result)
print("Minimum value:", f(result))
```

## Expected Output
```text
Optimal solution: 0.012845317982641
Minimum value: 0.000164986178350
```

Note: Output values vary on each run because the algorithm uses random initialization and random neighbor generation.
