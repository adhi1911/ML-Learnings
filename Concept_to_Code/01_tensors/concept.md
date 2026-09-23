# 01. Tensors & Memory Layout

## 1. What is the idea? (CONCEPT)
Tensors are multi-dimensional arrays with uniform data types. In PyTorch, a tensor wraps contiguous/strided memory with metadata: shape, stride, dtype, and device. Understanding shape transformations (view, reshape, transpose) and broadcasting prevents 90% of tensor bugs.

## 2. What is the math? (MATH)
A tensor $T \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_k}$.
Linear index mapping:
$$\text{index}(i_1, i_2, \dots, i_k) = \sum_{j=1}^k i_j \cdot \text{stride}_j$$
Broadcasting rule: trailing dimensions must match or be 1. Size 1 expands to match without copying data.

## 3. Shapes & Tensor Dimensions (SHAPES)
Input: `(B, T, D)` where B = Batch size, T = Sequence length / Time steps, D = Feature dim.
Transpose / Permute: `(B, T, D) -> (B, D, T)` changes strides; needs `.contiguous()` before `.view()`.

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch

# 1. Tensor creation & inspection
x = torch.randn(2, 3, 4)
print(f"Shape: {x.shape}, Stride: {x.stride()}, Dtype: {x.dtype}, Device: {x.device}")

# 2. Reshaping vs Transposing
# view requires contiguous memory; reshape handles non-contiguous by copying if needed
x_transposed = x.transpose(1, 2)  # shape (2, 4, 3)
print(f"Is contiguous after transpose: {x_transposed.is_contiguous()}")
x_flat = x_transposed.contiguous().view(2, -1)  # shape (2, 12)
print(f"Flattened shape: {x_flat.shape}")

# 3. Broadcasting demo
bias = torch.randn(4)  # shape (4,) -> broadcasts with (2, 3, 4)
y = x + bias
print(f"Broadcasted addition shape: {y.shape}")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Inspect `x.stride()` before and after `.transpose(0, 1)`.
2. Try calling `.view(-1)` on a transposed non-contiguous tensor and observe the `RuntimeError`.
3. Fix with `.contiguous().view(-1)` or `.reshape(-1)`.
4. Test broadcasting compatibility between shapes `(3, 1, 5)` and `(2, 5)`.
