# 07. Core Attention Mechanism

## 1. What is the idea? (CONCEPT)
Attention allows a sequence model to dynamically focus on relevant parts of an input sequence when processing a given token. It computes pairwise similarity between tokens and produces a weighted average of token representations.

## 2. What is the math? (MATH)
Given representations $X \in \mathbb{R}^{T \times d}$:
Similarity matrix:
$$S = X X^T \in \mathbb{R}^{T \times T}$$
Attention weights (row-wise softmax):
$$A = \text{softmax}(S, \text{dim}=-1)$$
Context representation:
$$Y = A X \in \mathbb{R}^{T \times d}$$

## 3. Shapes & Tensor Dimensions (SHAPES)
Input $X$: `(B, T, d)`
Similarity $S$: `(B, T, T)`
Attention weights $A$: `(B, T, T)` (each row sums to 1.0)
Output $Y$: `(B, T, d)`

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn.functional as F

B, T, d = 2, 4, 8
X = torch.randn(B, T, d)

# 1. Compute dot-product affinity between all pairs
affinity = torch.bmm(X, X.transpose(1, 2))  # (B, T, T)

# 2. Normalize rows with softmax to get probabilities
weights = F.softmax(affinity, dim=-1)       # (B, T, T)
print(f"Row sums (should be 1.0): {weights.sum(dim=-1)}")

# 3. Weighted aggregation of representations
out = torch.bmm(weights, X)                 # (B, T, d)
print(f"Input: {X.shape} -> Attention Output: {out.shape}")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Verify that uniform attention weights equal the simple average of tokens.
2. Observe how multiplying $S$ by a temperature factor controls peakiness/entropy of weights.
