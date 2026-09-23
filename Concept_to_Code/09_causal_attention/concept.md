# 09. Causal (Autoregressive) Attention

## 1. What is the idea? (CONCEPT)
In autoregressive language models (like GPT), tokens can only attend to past and current tokens, never future tokens. Causal attention achieves this by masking upper triangular scores with $-\infty$ before the softmax, forcing future attention probabilities to zero.

## 2. What is the math? (MATH)
Mask matrix $M \in \mathbb{R}^{T \times T}$:
$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

## 3. Shapes & Tensor Dimensions (SHAPES)
$Q, K, V$: `(B, T, d_k)`
Scores: `(B, T, T)`
Causal Mask: `(T, T)` lower triangular matrix (tril)
Attention Weights: `(B, T, T)` where entries $(i, j)$ with $j > i$ are exactly 0.0

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        
        # Register lower-triangular causal mask buffer (not a parameter)
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        # Compute Q, K, V in single matmul
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        scores = (q @ k.transpose(-2, -1)) / (C ** 0.5)  # (B, T, T)
        # Mask future positions with -inf
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        out = self.proj(attn @ v)
        return out, attn

x = torch.randn(2, 5, 16)
causal_attn = CausalSelfAttention(d_model=16, max_len=32)
out, weights = causal_attn(x)

print("First sample causal attention weights (row i only sees j <= i):")
print(weights[0].round(decimals=3))
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Confirm that row 0 has only 1 non-zero entry, row 1 has 2, etc.
2. Verify that modifying token at index $T-1$ does NOT change hidden states of tokens $0 \dots T-2$.
