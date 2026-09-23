# Concept to Code 🧪

> **"I can take an ML concept and implement it."**

This folder bridges the gap between theoretical ML/DL concepts and rock-solid PyTorch implementation.
This is your **laboratory**. It is designed for fast, focused iteration:

Each module answers **3 essential questions**:
1. **CONCEPT**: What is the idea?
2. **MATH**: What is the math formula and formal definition?
3. **PYTORCH**: How do I write it in clean PyTorch?

Plus:
- **SHAPES**: Exact tensor dimensions and batch flows.
- **EXPERIMENTS**: Immediate tests to break and inspect the mechanics.

---

## Roadmap

| # | Topic | Folder | Notebook |
|---|---|---|---|
| 01 | Tensors & Memory Layout | [`01_tensors/`](./01_tensors) | [`tensors.ipynb`](./01_tensors/tensors.ipynb) |
| 02 | Linear Layer (Affine Map) | [`02_linear_layer/`](./02_linear_layer) | [`linear_layer.ipynb`](./02_linear_layer/linear_layer.ipynb) |
| 03 | Loss & Gradients (Autograd) | [`03_loss_and_gradients/`](./03_loss_and_gradients) | [`loss_and_gradients.ipynb`](./03_loss_and_gradients/loss_and_gradients.ipynb) |
| 04 | Multi-Layer Perceptron (MLP) | [`04_mlp/`](./04_mlp) | [`mlp.ipynb`](./04_mlp/mlp.ipynb) |
| 05 | Token & Vector Embeddings | [`05_embeddings/`](./05_embeddings) | [`embeddings.ipynb`](./05_embeddings/embeddings.ipynb) |
| 06 | Bag of Words (BoW) | [`06_bow/`](./06_bow) | [`bow.ipynb`](./06_bow/bow.ipynb) |
| 07 | Core Attention Mechanism | [`07_attention/`](./07_attention) | [`attention.ipynb`](./07_attention/attention.ipynb) |
| 08 | Scaled Dot-Product (QKV) | [`08_qkv/`](./08_qkv) | [`qkv.ipynb`](./08_qkv/qkv.ipynb) |
| 09 | Causal (Autoregressive) Attention | [`09_causal_attention/`](./09_causal_attention) | [`causal_attention.ipynb`](./09_causal_attention/causal_attention.ipynb) |
| 10 | Full Transformer Block | [`10_transformer_block/`](./10_transformer_block) | [`transformer_block.ipynb`](./10_transformer_block/transformer_block.ipynb) |

---

## Recommended Learning Flow
1. Open the folder for your current concept.
2. Read `concept.md` or launch the `.ipynb` notebook.
3. Understand the input/output tensor shapes.
4. Run the PyTorch implementation cell.
5. Execute the **Experiment** section — alter tensor shapes, break assumptions, inspect gradients!
