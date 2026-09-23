# 02. Linear Layer (Affine Transformation)

## 1. What is the idea? (CONCEPT)
A linear (dense/affine) layer projects an input vector from dimension $d_{in}$ to $d_{out}$ using a weight matrix and an optional bias. It performs a rotation/scaling followed by translation.

## 2. What is the math? (MATH)
$$y = xW^T + b$$
Where:
- $x \in \mathbb{R}^{N \times d_{in}}$ (batch of vectors)
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$ (weights)
- $b \in \mathbb{R}^{d_{out}}$ (bias)
- $y \in \mathbb{R}^{N \times d_{out}}$

## 3. Shapes & Tensor Dimensions (SHAPES)
Input $x$: `(..., d_in)` — arbitrary leading batch dimensions allowed (e.g. `(B, T, d_in)`)
Weight $W$: `(d_out, d_in)`
Bias $b$: `(d_out,)`
Output $y$: `(..., d_out)` (e.g. `(B, T, d_out)`)

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn

# 1. Scratch Implementation
class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Kaiming uniform / standard init scale
        k = 1.0 / (in_features ** 0.5)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-k, k)) if bias else None

    def forward(self, x):
        # x: (..., in_features), weight.T: (in_features, out_features)
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

# 2. Compare with PyTorch nn.Linear
batch, seq_len, d_in, d_out = 4, 8, 16, 32
x = torch.randn(batch, seq_len, d_in)

custom_lin = ManualLinear(d_in, d_out)
torch_lin = nn.Linear(d_in, d_out)

# Copy weights to verify exact output equivalence
torch_lin.weight.data.copy_(custom_lin.weight.data)
torch_lin.bias.data.copy_(custom_lin.bias.data)

assert torch.allclose(custom_lin(x), torch_lin(x), atol=1e-6)
print("ManualLinear matches nn.Linear exactly!")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Measure number of parameters: $d_{in} \times d_{out} + d_{out}$.
2. Test passing 2D `(B, d_in)` vs 3D `(B, T, d_in)` vs 4D `(B, C, H, d_in)` into the layer.
3. Remove bias (`bias=False`) and inspect shape & parameters.
