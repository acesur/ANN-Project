# Theoretical Foundations of Artificial Neural Networks
## STW7088CEM - Advanced Implementation Study

### Abstract
This document provides the mathematical and theoretical foundations underlying our artificial neural network implementations for fraud detection, optical character recognition, and time series analysis. We present rigorous mathematical derivations, algorithmic foundations, and theoretical justifications for our architectural choices.

---

## 1. Introduction and Literature Review

### 1.1 Historical Context and Evolution

Artificial Neural Networks (ANNs) have evolved significantly since the pioneering work of McCulloch and Pitts (1943). The development trajectory includes:

- **Perceptron Era (1957-1969)**: Rosenblatt's perceptron and the subsequent limitations identified by Minsky and Papert (1969)
- **Renaissance Period (1986-present)**: Backpropagation algorithm by Rumelhart et al. (1986), leading to multi-layer networks
- **Deep Learning Revolution (2006-present)**: Hinton et al.'s deep belief networks and the current era of deep neural architectures

### 1.2 Current State of Research

#### 1.2.1 Fraud Detection with Neural Networks
Recent advances in financial fraud detection leverage deep learning architectures:

- **Varmedja et al. (2019)**: Demonstrated superior performance of deep neural networks over traditional machine learning in credit card fraud detection
- **Rtayli & Enneya (2020)**: Enhanced fraud detection using ensemble deep learning with feature selection
- **Zhang et al. (2021)**: Real-time fraud detection using LSTM networks with attention mechanisms

#### 1.2.2 Optical Character Recognition
Modern OCR systems have achieved near-human accuracy:

- **LeCun et al. (1998)**: Convolutional Neural Networks for handwritten digit recognition
- **Graves et al. (2009)**: Connectionist Temporal Classification for sequence labeling
- **Shi et al. (2017)**: CRNN architecture for scene text recognition

#### 1.2.3 Time Series Analysis with RNNs
Sequential data modeling has been revolutionized by recurrent architectures:

- **Hochreiter & Schmidhuber (1997)**: Long Short-Term Memory networks
- **Cho et al. (2014)**: Gated Recurrent Units for efficient sequence modeling
- **Vaswani et al. (2017)**: Transformer architecture with self-attention mechanisms

---

## 2. Mathematical Foundations

### 2.1 The Universal Approximation Theorem

**Theorem (Cybenko, 1989)**: Let φ be a continuous sigmoidal function. Then finite sums of the form:

```
G(x) = Σ(i=1 to N) αᵢφ(yᵢᵀx + θᵢ)
```

are dense in C(Iₙ), the space of continuous functions on the unit hypercube Iₙ.

**Implications**: This theorem guarantees that feedforward networks with a single hidden layer can approximate any continuous function to arbitrary accuracy, given sufficient neurons.

### 2.2 Backpropagation Algorithm

#### 2.2.1 Forward Pass
For a neural network with layers indexed by l:

```
z^(l) = W^(l)a^(l-1) + b^(l)
a^(l) = σ(z^(l))
```

Where:
- W^(l) ∈ ℝ^(n_l × n_{l-1}) is the weight matrix
- b^(l) ∈ ℝ^(n_l) is the bias vector
- σ is the activation function
- a^(l) is the activation vector

#### 2.2.2 Loss Function
For classification tasks, we use cross-entropy loss:

```
L(θ) = -Σ(i=1 to m) Σ(j=1 to k) y_ij log(ŷ_ij)
```

For regression tasks, we use mean squared error:

```
L(θ) = (1/2m) Σ(i=1 to m) ||y_i - ŷ_i||²
```

#### 2.2.3 Backward Pass (Gradient Computation)
The gradient computation follows the chain rule:

**Output layer error:**
```
δ^(L) = ∇_a L ⊙ σ'(z^(L))
```

**Hidden layer error:**
```
δ^(l) = ((W^(l+1))ᵀδ^(l+1)) ⊙ σ'(z^(l))
```

**Parameter gradients:**
```
∇_{W^(l)} L = δ^(l)(a^(l-1))ᵀ
∇_{b^(l)} L = δ^(l)
```

### 2.3 Activation Functions

#### 2.3.1 Sigmoid Function
```
σ(x) = 1/(1 + e^(-x))
σ'(x) = σ(x)(1 - σ(x))
```

**Properties:**
- Range: (0, 1)
- Smooth, differentiable
- Suffers from vanishing gradient problem

#### 2.3.2 Rectified Linear Unit (ReLU)
```
ReLU(x) = max(0, x)
ReLU'(x) = {1 if x > 0, 0 if x ≤ 0}
```

**Properties:**
- Computationally efficient
- Helps mitigate vanishing gradient problem
- Can suffer from "dead neurons"

#### 2.3.3 Hyperbolic Tangent (Tanh)
```
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

**Properties:**
- Range: (-1, 1)
- Zero-centered
- Still suffers from vanishing gradient problem

---

## 3. Optimization Algorithms

### 3.1 Gradient Descent Variants

#### 3.1.1 Stochastic Gradient Descent (SGD)
```
θ_{t+1} = θ_t - α∇_θ L(θ_t)
```

Where α is the learning rate.

#### 3.1.2 Adam Optimizer
Combines momentum and adaptive learning rates:

```
m_t = β₁m_{t-1} + (1-β₁)∇_θ L(θ_t)
v_t = β₂v_{t-1} + (1-β₂)(∇_θ L(θ_t))²

m̂_t = m_t/(1-β₁ᵗ)
v̂_t = v_t/(1-β₂ᵗ)

θ_{t+1} = θ_t - α(m̂_t/(√v̂_t + ε))
```

**Parameters:**
- β₁ = 0.9 (momentum decay)
- β₂ = 0.999 (squared gradient decay)
- ε = 1e-8 (numerical stability)

### 3.2 Regularization Techniques

#### 3.2.1 L2 Regularization (Weight Decay)
```
L_reg = L + λ/2 Σ||W^(l)||_F²
```

Where λ is the regularization parameter and ||·||_F is the Frobenius norm.

#### 3.2.2 Dropout
During training, randomly set neurons to zero with probability p:
```
h_dropout = h ⊙ mask
```
where mask ~ Bernoulli(1-p)

---

## 4. Specialized Architectures

### 4.1 Convolutional Neural Networks (CNNs)

#### 4.1.1 Convolution Operation
For input X and filter W:
```
(X * W)_{i,j} = Σ_m Σ_n X_{i+m,j+n} W_{m,n}
```

#### 4.1.2 Pooling Operation
Max pooling:
```
pool(X)_{i,j} = max_{m,n∈R_{i,j}} X_{m,n}
```

### 4.2 Recurrent Neural Networks (RNNs)

#### 4.2.1 Vanilla RNN
```
h_t = tanh(W_h h_{t-1} + W_x x_t + b)
y_t = W_y h_t + b_y
```

#### 4.2.2 Long Short-Term Memory (LSTM)
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)  # Hidden state
```

#### 4.2.3 Gated Recurrent Unit (GRU)
```
z_t = σ(W_z · [h_{t-1}, x_t])  # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])  # Reset gate
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])  # Candidate activation
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # Final activation
```

---

## 5. Evaluation Metrics and Statistical Analysis

### 5.1 Classification Metrics

#### 5.1.1 Confusion Matrix Analysis
For binary classification:
```
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1-Score = 2 * (Precision * Recall)/(Precision + Recall)
```

#### 5.1.2 ROC-AUC Score
Area Under the Receiver Operating Characteristic Curve:
```
AUC = ∫₀¹ TPR(FPR⁻¹(x))dx
```

### 5.2 Regression Metrics

#### 5.2.1 Mean Squared Error
```
MSE = (1/n)Σᵢ₌₁ⁿ(yᵢ - ŷᵢ)²
```

#### 5.2.2 R-squared
```
R² = 1 - SS_res/SS_tot
```
where SS_res = Σ(yᵢ - ŷᵢ)² and SS_tot = Σ(yᵢ - ȳ)²

---

## 6. Theoretical Justifications for Our Implementations

### 6.1 Task 1: Fraud Detection
- **Architecture Choice**: Deep feedforward networks with dropout
- **Theoretical Basis**: High-dimensional feature spaces require non-linear decision boundaries
- **Loss Function**: Binary cross-entropy for probability calibration

### 6.2 Task 2: MNIST OCR
- **Architecture Choice**: Convolutional Neural Networks
- **Theoretical Basis**: Translation invariance and local connectivity assumptions
- **Progressive Feature Learning**: Hierarchical feature extraction

### 6.3 Task 3: Time Series Analysis
- **Architecture Choice**: LSTM/GRU networks with attention
- **Theoretical Basis**: Long-range dependencies in temporal data
- **Memory Mechanisms**: Explicit control over information flow

---

## References

1. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2(4), 303-314.

2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.

3. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.

5. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.