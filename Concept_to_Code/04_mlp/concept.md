# 04. Multi-Layer Perceptron (MLP)

## 1. What is the idea? (CONCEPT)
A single linear layer can only learn linear decision boundaries. Stacking multiple linear layers with non-linear activation functions (ReLU, GELU, SiLU) allows the network to approximate arbitrary continuous functions (Universal Approximation Theorem).

## 2. What is the math? (MATH)
$$h_1 = \sigma(x W_1^T + b_1)$$
$$h_2 = \sigma(h_1 W_2^T + b_2)$$
$$\hat{y} = h_2 W_3^T + b_3$$
Where $\sigma$ is a non-linear activation function, e.g. $\text{ReLU}(z) = \max(0, z)$ or $\text{GELU}(z) = z \Phi(z)$.

## 3. Shapes & Tensor Dimensions (SHAPES)
Input $x$: `(B, d_in)`
Hidden 1 $h_1$: `(B, d_hidden)`
Hidden 2 $h_2$: `(B, d_hidden)`
Output $\hat{y}$: `(B, d_out)`

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate and verify forward pass
B, d_in, d_hidden, d_out = 16, 64, 128, 10
mlp = MLP(d_in, d_hidden, d_out)
x = torch.randn(B, d_in)
out = mlp(x)

print(f"MLP Input: {x.shape} -> Output: {out.shape}")
total_params = sum(p.numel() for p in mlp.parameters())
print(f"Total Parameters: {total_params}")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Remove the activation functions (`GELU`) and show mathematically & empirically that the multi-layer network collapses to a single linear map.
2. Train on XOR data (`(4, 2)`) to see single layer fail and MLP succeed.
