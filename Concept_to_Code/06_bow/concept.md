# 06. Bag of Words (BoW)

## 1. What is the idea? (CONCEPT)
Bag of Words represents a document as an unordered collection of words, ignoring grammar and word order but keeping count frequency. It forms a fixed-size vector representation of variable-length texts.

## 2. What is the math? (MATH)
Given vocabulary $V = \{w_1, w_2, \dots, w_{|V|}\}$:
$$X_{i, j} = \sum_{t \in \text{doc}_i} \mathbb{I}(t == w_j)$$
Or normalized frequency:
$$\tilde{X}_{i, j} = \frac{X_{i, j}}{\sum_k X_{i, k}}$$

## 3. Shapes & Tensor Dimensions (SHAPES)
Input documents: list of $N$ token ID lists of variable length $L_i$
BoW matrix: `(N, vocab_size)`
Output after Linear classifier: `(N, num_classes)`

## 4. How do I write it in PyTorch? (PYTORCH)
```python
import torch
import torch.nn as nn

def texts_to_bow(tokenized_docs, vocab_size):
    # tokenized_docs: list of lists of token indices
    N = len(tokenized_docs)
    bow = torch.zeros(N, vocab_size)
    for i, doc in enumerate(tokenized_docs):
        for token_id in doc:
            bow[i, token_id] += 1.0
    return bow

# Sample docs and vocabulary
vocab = {"cat": 0, "sat": 1, "on": 2, "mat": 3, "dog": 4, "bark": 5}
docs = [
    [0, 1, 2, 3],       # "cat sat on mat"
    [4, 5],             # "dog bark"
    [0, 1, 0, 1]        # "cat sat cat sat"
]

bow_matrix = texts_to_bow(docs, vocab_size=len(vocab))
print("BoW Matrix:")
print(bow_matrix)

# Simple BoW Classifier
classifier = nn.Linear(len(vocab), 2)  # binary sentiment / topic
logits = classifier(bow_matrix)
print(f"Logits shape: {logits.shape}")
```

## 5. Experiments & Exercises (EXPERIMENT)
1. Compare raw counts vs TF (normalized by document length) vs TF-IDF weighting.
2. Note failure of BoW to distinguish 'not good, very bad' from 'not bad, very good'.
