# 10. Transformer Block

## 1. What is the idea? (CONCEPT)
A standard Transformer decoder block (pre-LN GPT style) connects Causal Multi-Head Self-Attention and a position-wise Feed-Forward Network (MLP) using residual connections and Layer Normalization. Residual connections allow gradients to flow unimpeded through hundreds of layers.

## 2. What is the math? (MATH)
$$x_1 = x + \text{Attention}(\text{LayerNorm}(x))$$
$$x_2 = x_1 + \text{FFN}(\text{LayerNorm}(x_1))$$
Where:
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
$$\text{FFN}(z) = \text{GELU}(z W_1 + b_1) W_2 + b_2$$

## 3. Shapes & Tensor Dimensions (SHAPES)
Input $x$: `(B, T, d_model)`
After Self-Attention + Residual: `(B, T, d_model)`
After FFN + Residual: `(B, T, d_model)`
(Shapes are preserved throughout the block)

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SimpleCausalAttention(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return self.proj(attn @ v)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SimpleCausalAttention(d_model, max_len=max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x):
        # Pre-LN architecture (modern GPT style)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# Test Block
B, T, d_model = 4, 16, 64
x = torch.randn(B, T, d_model)
block = TransformerBlock(d_model)
out = block(x)

print(f"Block Input: {x.shape} -> Block Output: {out.shape}")
assert out.shape == x.shape
print("Transformer block forward pass successful!")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Stack $N=4$ blocks and add token + position embeddings to create a miniature GPT.
2. Remove residual additions (`x + ...`) and observe gradient propagation during training.
