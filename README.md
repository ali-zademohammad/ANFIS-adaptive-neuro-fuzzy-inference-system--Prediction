# ANFIS Implementation for Viscosity Prediction


This repository contains an implementation of the **Adaptive Neuro-Fuzzy Inference System (ANFIS)** to predict the viscosity of a fluid based on temperature and pressure inputs. Two distinct methods are provided for ANFIS implementation:

**Hybrid Optimization Approach**: This method uses a combination of fuzzy logic with linear regression and optimization to predict viscosity.       
**PyTorch-based Deep Learning Approach**: This method leverages a neural network with Gaussian membership functions to model the relationship between inputs (temperature and pressure) and viscosity.   

Both methods use an example dataset to demonstrate the models and make predictions. The dataset contains temperature, pressure, and viscosity values for a fluid.   

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

1. **Input Layer**: Receives raw input values (temperature and pressure).
2. **Fuzzification Layer**: Converts inputs into fuzzy values using Gaussian membership functions.
3. **Rule Layer**: Evaluates firing strengths of fuzzy rules using the product T-norm.
4. **Normalization Layer**: Normalizes firing strengths across all rules.
5. **Defuzzification Layer**: Computes the final output using a weighted sum of rule contributions.

## Implementation Details
The implementation consists of several key components:

1. **Initialization**:
   - **K-Means Clustering** is used to determine initial membership function centers.
   - Sigma (\(\sigma\)) values are calculated based on the distance between cluster centers.

   - Uses K-Means clustering to initialize membership function centers
   - Calculates initial sigma values based on cluster centers

3. **Membership Functions**:
   - Gaussian membership functions are used for fuzzification
   - - Represents fuzzy sets for temperature (\(T\)) and pressure (\(P\)).
   - Parameters (c, σ) are optimized during training

4. **Rule Evaluation**:
   - Four rules combining temperature and pressure conditions
   - Rule firing strengths calculated using product T-norm
   - - Defines fuzzy rules based on combinations of low/high fuzzy sets for \(T\) and \(P\).
   - Calculates firing strengths using the product of membership values.

4. **Parameter Estimation**:
   - Uses **linear regression** to estimate rule coefficients (\(p, q, r\)).
   - Refines parameters through **L-BFGS-B optimization** to minimize prediction error.


5. **Hybrid Learning**:
   - Initial parameters estimated using linear regression
   - Final optimization using L-BFGS-B method

6. **Prediction**:
   - Combines rule outputs using weighted average
   - Produces final viscosity prediction
---

### PyTorch-Based Neural Network Approach
1. **Learnable Membership Functions**:
   - Membership function parameters (\(c, \sigma\)) are initialized randomly and updated during training.

2. **Rule Computation**:
   - Firing strengths are calculated for each rule using Gaussian membership functions.

3. **End-to-End Training**:
   - A neural network computes and combines rule outputs.
   - Uses **Adam optimizer** and **MSE loss** for training.

---


## Mathematical Background

### Gaussian Membership Function
The Gaussian function is used to represent fuzzy sets:  

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
