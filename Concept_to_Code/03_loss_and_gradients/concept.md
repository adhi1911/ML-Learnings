# 03. Loss & Gradients (Autograd Mechanics)

## 1. What is the idea? (CONCEPT)
Training is guided by a scalar loss measuring error between prediction and ground truth. PyTorch's autograd engine builds a directed acyclic computational graph during the forward pass. Calling `loss.backward()` traverses this graph backward via the chain rule to accumulate partial derivatives into parameter `.grad` attributes.

## 2. What is the math? (MATH)
Chain Rule:
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$$
Mean Squared Error (MSE):
$$L_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
Cross Entropy (with logits $z$):
$$L_{CE} = -\log \left( \frac{e^{z_y}}{\sum_j e^{z_j}} \right)$$

## 3. Shapes & Tensor Dimensions (SHAPES)
Predictions $\hat{y}$: `(N, C)` logits
Targets $y$: `(N,)` class indices (for CrossEntropy) or `(N, C)` (for MSE)
Loss $L$: scalar `()` (or shape `(1,)`)
Gradients $W.grad$: identical shape to $W$ `(C, D)`

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Forward & Backward walkthrough
x = torch.randn(8, 4)
target = torch.randint(0, 3, (8,))  # 3 classes

w = torch.randn(3, 4, requires_grad=True)
b = torch.zeros(3, requires_grad=True)

# 1. Forward pass
logits = x @ w.T + b
loss = F.cross_entropy(logits, target)
print(f"Loss: {loss.item():.4f}")

# 2. Backward pass
loss.backward()

print(f"w.grad shape: {w.grad.shape}, matches w: {w.grad.shape == w.shape}")
print(f"b.grad shape: {b.grad.shape}, matches b: {b.grad.shape == b.shape}")

# 3. Manual gradient descent step
lr = 0.05
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
    # Gradients accumulate by default; must zero them out before next backward
    w.grad.zero_()
    b.grad.zero_()
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Run `loss.backward()` twice without `zero_()` — observe gradient doubling.
2. Inspect `logits.grad_fn` to see the backward hook attached by autograd.
3. Compare manual MSE loss `((pred - target)**2).mean()` with `nn.MSELoss()`.
