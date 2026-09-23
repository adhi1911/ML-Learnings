# 08. Query, Key, Value (Scaled Dot-Product)

## 1. What is the idea? (CONCEPT)
Instead of using raw token representations directly, Scaled Dot-Product Attention projects representations into three distinct spaces: Queries (what am I looking for?), Keys (what do I offer?), and Values (what information do I contain?). Scaling by $\sqrt{d_k}$ prevents vanishing gradients in the softmax.

## 2. What is the math? (MATH)
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
Where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$.

## 3. Shapes & Tensor Dimensions (SHAPES)
Input $X$: `(B, T, d_model)`
$Q$: `(B, T, d_k)`
$K$: `(B, T, d_k)`
$V$: `(B, T, d_v)`
Scores $Q K^T / \sqrt{d_k}$: `(B, T, T)`
Output: `(B, T, d_v)`

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_v, bias=False)
        self.scale = d_k ** 0.5

    def forward(self, x):
        # B = batch, T = seq_len
        q = self.w_q(x)  # (B, T, d_k)
        k = self.w_k(x)  # (B, T, d_k)
        v = self.w_v(x)  # (B, T, d_v)

        # Scaled dot-product
        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, T, T)
        attn_weights = F.softmax(scores, dim=-1)          # (B, T, T)
        out = attn_weights @ v                            # (B, T, d_v)
        return out, attn_weights

B, T, d_model, d_k, d_v = 2, 6, 32, 16, 16
x = torch.randn(B, T, d_model)
attn = ScaledDotProductAttention(d_model, d_k, d_v)
out, weights = attn(x)

print(f"Output shape: {out.shape}, Weights shape: {weights.shape}")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Remove the $1/\sqrt{d_k}$ scaling factor for large $d_k = 512$ and observe softmax saturation (outputs become one-hot, gradients vanish).
2. Inspect learned attention matrix for diagonal dominance.
