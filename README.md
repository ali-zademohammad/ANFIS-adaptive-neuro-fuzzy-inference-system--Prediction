# ANFIS Implementation for Viscosity Prediction

This repository contains a Python implementation of an Adaptive Neuro-Fuzzy Inference System (ANFIS) for predicting viscosity based on temperature and pressure inputs. The implementation follows the standard ANFIS architecture with hybrid learning.

## Table of Contents
1. [Introduction](#introduction)
2. [ANFIS Architecture](#anfis-architecture)
3. [Implementation Details](#implementation-details)
4. [Mathematical Background](#mathematical-background)
5. [Usage](#usage)
6. [Dependencies](#dependencies)
7. [References](#references)

## Introduction
ANFIS (Adaptive Neuro-Fuzzy Inference System) is a powerful hybrid intelligent system that combines the learning capabilities of neural networks with the interpretability of fuzzy systems. This implementation focuses on predicting viscosity (V) based on temperature (T) and pressure (P) inputs.

## ANFIS Architecture
The system follows a 5-layer architecture:

1. **Input Layer**: Receives temperature and pressure values
2. **Fuzzification Layer**: Applies Gaussian membership functions
3. **Rule Layer**: Computes firing strengths of fuzzy rules
4. **Normalization Layer**: Normalizes rule strengths
5. **Defuzzification Layer**: Produces final output using linear regression

## Implementation Details
The implementation consists of several key components:

1. **Initialization**:
   - Uses K-Means clustering to initialize membership function centers
   - Calculates initial sigma values based on cluster centers

2. **Membership Functions**:
   - Gaussian membership functions are used for fuzzification
   - Parameters (c, σ) are optimized during training

3. **Rule Evaluation**:
   - Four rules combining temperature and pressure conditions
   - Rule firing strengths calculated using product T-norm

4. **Hybrid Learning**:
   - Initial parameters estimated using linear regression
   - Final optimization using L-BFGS-B method

5. **Prediction**:
   - Combines rule outputs using weighted average
   - Produces final viscosity prediction

## Mathematical Background
### Gaussian Membership Function
```math
\mu(x) = e^{-\frac{(x - c)^2}{2\sigma^2}}
```
where:
- c: center of the membership function
- σ: width of the membership function

### Rule Firing Strength
```math
w_i = \mu_{T}(T) \times \mu_{P}(P)
```

### Output Calculation
```math
y = \frac{\sum_{i=1}^{n} w_i \cdot (p_i T + q_i P + r_i)}{\sum_{i=1}^{n} w_i}
```

## Usage
1. Prepare your dataset with columns: Temperature, Pressure, Viscosity
2. Import the required libraries:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.optimize import minimize
```

3. Initialize the ANFIS model:
```python
# Initialize membership functions
c_T, sigma_T = initialize_membership_functions(T)
c_P, sigma_P = initialize_membership_functions(P)
```

4. Train the model:
```python
# Calculate rule firing strengths
rules = firing_strength(T, P, c_T, sigma_T, c_P, sigma_P)

# Optimize parameters
result = minimize(loss_function, x0=np.hstack([params_initial, np.zeros(4)]), method='L-BFGS-B')
params_optimized = result.x
```

5. Make predictions:
```python
predictions = anfis_predict(params_optimized, T, P, rules)
```

## Dependencies
- Python 3.7+
- NumPy
- scikit-learn
- SciPy

Install requirements:
```bash
pip install numpy scikit-learn scipy
```

## References
1. Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-based Fuzzy Inference System. IEEE Transactions on Systems, Man, and Cybernetics.
2. Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. IEEE Transactions on Systems, Man, and Cybernetics.
3. K-means clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
4. L-BFGS-B optimization: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
