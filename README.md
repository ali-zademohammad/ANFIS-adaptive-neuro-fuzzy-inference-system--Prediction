# ANFIS Implementation for Viscosity Prediction

This repository contains an implementation of the **Adaptive Neuro-Fuzzy Inference System (ANFIS)** to predict the viscosity of a fluid based on temperature and pressure inputs. Two distinct methods are provided for ANFIS implementation:

1. **Hybrid Optimization Approach**: This method uses a combination of fuzzy logic with linear regression and optimization to predict viscosity.
2. **PyTorch-based Deep Learning Approach**: This method leverages a neural network with Gaussian membership functions to model the relationship between inputs (temperature and pressure) and viscosity.

Both methods use an example dataset to demonstrate the models and make predictions. The dataset contains temperature, pressure, and viscosity values for a fluid.    
<div style="text-align: center;">
    <img src="https://www.mdpi.com/mca/mca-22-00043/article_deploy/html/images/mca-22-00043-g003-550.jpg" alt="image" style="width: 80%; height: auto;">
</div>


## Table of Contents
1. [Introduction](#introduction)
2. [ANFIS Architecture](#anfis-architecture)
3. [Implementation Details](#implementation-details)
   - [Hybrid Optimization Approach](#hybrid-optimization-approach)
   - [PyTorch-based Deep Learning Approach](#pytorch-based-deep-learning-approach)
4. [Mathematical Background](#mathematical-background)
5. [Dataset](#dataset)
6. [Usage](#usage)
   - [Hybrid Optimization Approach](#usage-hybrid-optimization-approach)
   - [PyTorch-based Deep Learning Approach](#usage-pytorch-based-deep-learning-approach)
7. [Dependencies](#dependencies)
8. [File Structure](#file-structure)
9. [Contributions](#contributions)
10. [References](#references)

## Introduction
ANFIS (Adaptive Neuro-Fuzzy Inference System) is a hybrid intelligent system that combines the learning capabilities of neural networks with the interpretability of fuzzy systems. This implementation focuses on predicting viscosity (V) based on temperature (T) and pressure (P) inputs.

## ANFIS Architecture
### Hybrid Optimization Approach
The system follows a five-layer architecture:

1. **Input Layer**: Receives raw input values (temperature and pressure).
2. **Fuzzification Layer**: Converts inputs into fuzzy values using Gaussian membership functions.
3. **Rule Layer**: Evaluates firing strengths of fuzzy rules using the product T-norm.
4. **Normalization Layer**: Normalizes firing strengths across all rules.
5. **Defuzzification Layer**: Computes the final output using a weighted sum of rule contributions.

### PyTorch-based Deep Learning Approach
The architecture for this approach extends the hybrid optimization model with:
- Learnable Gaussian membership functions with trainable centers and widths.
- Rule-based modeling for interpretability.
- End-to-end training using backpropagation in PyTorch.

## Implementation Details
### Hybrid Optimization Approach
1. **Initialization**:
   - Uses K-Means clustering to determine initial membership function centers.
   - Calculates initial sigma (σ) values based on the distance between cluster centers.

2. **Membership Functions**:
   - Gaussian membership functions represent fuzzy sets for temperature (μ_T) and pressure (μ_P).
   - Parameters (center, sigma) are optimized during training.

3. **Rule Evaluation**:
   - Defines fuzzy rules based on combinations of low/high fuzzy sets for \(T\) and \(P\).
   - Calculates rule firing strengths using the product of membership values.

4. **Parameter Estimation**:
   - Uses **linear regression** to estimate rule coefficients (\(p, q, r\)).
   - Refines parameters through **L-BFGS-B optimization** to minimize prediction error.

5. **Prediction**:
   - Combines rule outputs using weighted averages to produce viscosity predictions.

### PyTorch-Based Deep Learning Approach
1. **Learnable Membership Functions**:
   - Membership function parameters (\(c, σ\)) are initialized randomly and updated during training.

2. **Rule Computation**:
   - Firing strengths are calculated for each rule using Gaussian membership functions.

3. **End-to-End Training**:
   - A neural network computes and combines rule outputs.
   - Uses **Adam optimizer** and **MSE loss** for training.

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


## Dataset

| Temperature (T) | Pressure (P) | Viscosity (V) |
|------------------|--------------|---------------|
| 50              | 1.0          | 120           |
| 60              | 1.2          | 125           |
| 70              | 1.5          | 130           |
| 80              | 1.7          | 135           |
| 90              | 2.0          | 140           |

## Usage
### Hybrid Optimization Approach
1. Prepare your dataset with columns: Temperature, Pressure, Viscosity.
2. Import required libraries:
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
result = minimize(loss_function, x0=params_initial, method='L-BFGS-B')
params_optimized = result.x
```
5. Make predictions:
```python
predictions = anfis_predict(params_optimized, T, P, rules)
```

### PyTorch-based Deep Learning Approach
1. Install dependencies:
```bash
pip install torch numpy
```

2. Train the model:
```python
# Initialize and train the model
model = ANFIS(num_rules=4, input_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    predictions = model(data)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```
3. Predict new data:
```python
x_new = torch.tensor([[75.0, 1.6]], dtype=torch.float32)
predicted_viscosity = model(x_new)
print(f"Predicted Viscosity for T=75, P=1.6: {predicted_viscosity.item()}")
```

## Example Output

```
Epoch 0, Loss: 25.46
Epoch 100, Loss: 0.15
...
Epoch 900, Loss: 0.01
Predicted Viscosity for T=75, P=1.6: 132.4
```

## Dependencies
- Python 3.7+
- NumPy
- scikit-learn
- SciPy
- PyTorch

Install all dependencies with:
```bash
pip install numpy scikit-learn scipy torch
```

## File Structure
- **Hybrid Optimization Approach**: [ANFIS Hybrid Implementation](https://github.com/ali-zademohammad/ANFIS-adaptive-neuro-fuzzy-inference-system-Prediction/blob/221591c21af89e9a561d905c0b00e3dcdf3d621d/ANFIS-pytorch.ipynb)   
- **PyTorch-based Deep Learning Approach**: [ANFIS PyTorch Implementation](https://github.com/ali-zademohammad/ANFIS-adaptive-neuro-fuzzy-inference-system--Prediction/blob/221591c21af89e9a561d905c0b00e3dcdf3d621d/ANFIS-pytorch.ipynb)  
- **Documentation**: [README.md](https://github.com/ali-zademohammad/ANFIS-adaptive-neuro-fuzzy-inference-system--Prediction/blob/221591c21af89e9a561d905c0b00e3dcdf3d621d/README.md)

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests.

## References
1. Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-based Fuzzy Inference System. IEEE Transactions on Systems, Man, and Cybernetics.
2. Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. IEEE Transactions on Systems, Man, and Cybernetics.
3. K-means clustering: [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
4. L-BFGS-B optimization: [SciPy Minimize L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
