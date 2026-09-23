# Machine Learning Laboratory 🧠🔬

Welcome to the **ML-Learnings** repository — an active implementation laboratory for machine learning, deep learning, and transformer architectures.

This repository tracks hands-on implementation progress from core mathematical foundations up through modern transformer architectures and production applications.

---

## 📂 Repository Map

```text
ML-Learnings/
│
├── Concept_to_Code/              # 🧪 Active Daily Lab: Core concepts implemented from scratch
│   ├── 01_tensors/               # Strides, memory layout, broadcasting
│   ├── 02_linear_layer/          # Manual affine map vs nn.Linear
│   ├── 03_loss_and_gradients/    # Autograd engine, graph traversal, backprop
│   ├── 04_mlp/                   # Non-linear activations, universal approximation
│   ├── 05_embeddings/            # Token lookup & vector projections
│   ├── 06_bow/                   # Bag of Words & text representation
│   ├── 07_attention/             # Dot-product similarity & row softmax
│   ├── 08_qkv/                   # Scaled dot-product (Q, K, V projections)
│   ├── 09_causal_attention/      # Autoregressive causal triangular masking
│   └── 10_transformer_block/     # Pre-LN decoder block + residual connections
│
├── 00_Foundations/               # 📐 Mathematical & tensor fundamentals, EDA, preprocessing
│   ├── tensors.ipynb             # Tensor manipulations
│   ├── series_tensors.ipynb      # PyTorch tensor basics
│   ├── autograd.ipynb            # Gradient computation & autograd
│   ├── multivariate_analysis.ipynb # Exploratory data analysis
│   ├── normalization_practice.ipynb # Feature scaling
│   ├── standardization_practice.ipynb # Z-score scaling
│   ├── building_models.ipynb     # Module compositions
│   └── training.ipynb            # Standard optimization loop
│
├── 01_Models_From_Scratch/       # 🛠️ Raw implementations without black-box modules
│   ├── perceptron_scratch.ipynb  # Single-layer perceptron
│   ├── my_neural_network.ipynb   # Multi-layer neural network from scratch
│   ├── logistic_regression.ipynb # Logistic regression classifier
│   ├── activation_functions.py   # Sigmoid, ReLU, Tanh, Softmax
│   └── loss_functions.py         # MSE, CrossEntropy, BinaryCrossEntropy
│
├── 02_PyTorch/                   # 🔥 PyTorch modules, workflows, and computer vision
│   ├── intro.ipynb               # PyTorch workflow overview
│   ├── modelling.ipynb           # Model architecture patterns
│   ├── linear_regression.ipynb   # nn.Linear optimization
│   ├── lenet_cnn.py              # Classical LeNet-5 CNN
│   └── image_classification.ipynb # Image classification pipeline
│
├── 03_Transformers/              # ⚡ Modern Transformer & Language Model architectures
│   ├── gpt_from_scratch/         # Karpathy Andrej GPT walkthrough (gpt.py, gpt-dev.ipynb)
│   └── finetuning/               # BERT & transformer fine-tuning for classification
│
├── Projects/                     # 🚀 Applied ML & End-to-End Projects
│   ├── basic_ml_projects/        # Car price predictor, breast cancer, rock vs mine
│   ├── customer_churn_ann.ipynb  # ANN customer churn prediction
│   └── rag_application/          # Retrieval Augmented Generation system
│
├── data/                         # 📊 Centralized datasets (CSV & archive data)
│
└── archive/                      # 📦 Preserved historical course scrapers, books, and scratchpads
    ├── hands_on_ml_book/         # Aurélien Géron book notebooks
    ├── llm_course/               # Legacy LLM course scratchpads
    ├── series_notebook_misc/     # Miscellaneous lecture notes
    ├── dump/                     # Ad-hoc experimental scripts
    └── scratch_notebooks/        # Dataset imports & quick inspections
```

---

## 🧭 How to Navigate

1. **For Concept Mastery**: Head into [`Concept_to_Code/`](./Concept_to_Code/). Every folder contains a `concept.md` and an executable `.ipynb` covering:
   - **Concept**: Core theoretical intuition.
   - **Math**: Exact mathematical formulation.
   - **Shapes**: Input/output tensor shapes (`(B, T, D)`).
   - **PyTorch**: Clean, idiomatic PyTorch implementation.
   - **Experiment**: Concrete inspection tasks and stress tests.

2. **For Karpathy GPT Work**: Head into [`03_Transformers/gpt_from_scratch/`](./03_Transformers/gpt_from_scratch/).

3. **For Datasets**: All project datasets are centralized in [`data/`](./data/).
