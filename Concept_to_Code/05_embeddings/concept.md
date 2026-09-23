# 05. Token & Vector Embeddings

## 1. What is the idea? (CONCEPT)
Discrete tokens (words, subwords, characters) cannot be fed directly into neural networks. An embedding layer maps discrete integer token IDs into dense continuous vector representations. In practice, `nn.Embedding` is an efficient lookup table mathematically equivalent to multiplying a one-hot vector by a weight matrix.

## 2. What is the math? (MATH)
Weight matrix $E \in \mathbb{R}^{V \times d}$ where $V$ is vocabulary size and $d$ is embedding dimension.
For token index $i$:
$$\text{Embed}(i) = E[i, :] = \mathbf{e}_i E$$
where $\mathbf{e}_i$ is the $1 \times V$ one-hot vector.

## 3. Shapes & Tensor Dimensions (SHAPES)
Input token IDs: `(B, T)` integer tensor $\in [0, V-1]$
Embedding matrix: `(V, d)`
Output embeddings: `(B, T, d)`

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 1000
embed_dim = 64

# 1. nn.Embedding lookup
emb_layer = nn.Embedding(vocab_size, embed_dim)
tokens = torch.tensor([[10, 42, 99], [3, 10, 500]])  # shape (2, 3)
dense_vecs = emb_layer(tokens)
print(f"Token IDs: {tokens.shape} -> Embedded Vectors: {dense_vecs.shape}")

# 2. Prove equivalence with One-Hot matrix multiplication
one_hot = F.one_hot(tokens, num_classes=vocab_size).float()  # (2, 3, 1000)
matmul_vecs = one_hot @ emb_layer.weight  # (2, 3, 64)

assert torch.allclose(dense_vecs, matmul_vecs)
print("Lookup is mathematically identical to one_hot @ weight!")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Measure execution speed and memory between `nn.Embedding(10000, 256)` lookup vs full `one_hot @ weight`.
2. Implement sinusoidal positional embeddings and add them elementwise to token embeddings.
