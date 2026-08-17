# Swin Transformer from Scratch — CIFAR-10 Image Classification

This repository contains a **from-scratch PyTorch implementation of the Swin Transformer** (Shifted Window Transformer), built as part of a paper-review / reimplementation exercise. Instead of using a pre-built library, every core component of the architecture — window-based self-attention, relative position bias, shifted windows, and patch merging — is implemented manually and trained end-to-end on the **CIFAR-10** dataset.

## 📄 Reference Paper

This project is a reimplementation of the ideas introduced in:

> **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
> Liu, Ze, et al. (2021)

The Swin Transformer improves on the standard Vision Transformer (ViT) by computing self-attention within local, non-overlapping windows instead of globally over the whole image. Windows are shifted between consecutive blocks so that information can flow across window boundaries, and a patch-merging step progressively builds a hierarchical (multi-scale) feature representation — similar in spirit to how CNNs build feature pyramids.

## 🧠 Architecture Overview

The model implemented in this notebook follows the core building blocks of Swin Transformer:

- **Patch Embedding** — splits the input image into small patches using a strided convolution.
- **Window Attention** — computes multi-head self-attention restricted to local windows, with a learnable **relative position bias** added to the attention scores.
- **Shifted Window Mechanism** — alternates between regular and cyclically-shifted window partitions (with an attention mask) so consecutive blocks can exchange information across window boundaries.
- **Swin Transformer Block** — combines window attention with a pre-norm residual structure and an MLP (GELU) feed-forward block.
- **Patch Merging** — downsamples the spatial resolution by 2× while doubling the channel dimension, forming a hierarchical, multi-stage backbone.
- **Classification Head** — a final layer norm, global average pooling, and a linear layer producing the class logits.

The network is organized into **3 stages** with increasing channel dimensions and decreasing spatial resolution, followed by a classification head for the 10 CIFAR-10 classes.

## 🗂️ Dataset

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (60,000 32×32 color images across 10 classes)
- **Training augmentations:** random crop with padding, random horizontal flip, normalization
- **Test preprocessing:** normalization only

## ⚙️ Training Setup

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 0.05 |
| LR schedule | Cosine Annealing |
| Loss function | Cross-Entropy Loss |
| Epochs | 20 |
| Batch size | 128 |
| Device | CUDA / MPS / CPU (auto-detected) |

## 📊 Results

### Training vs Test Accuracy

The model was trained for 20 epochs, reaching a final test accuracy of approximately **77–78%** on CIFAR-10.

![Training vs Test Accuracy](Ekran%20Resmi%202026-08-17%2018.17.20.png)


### Confusion Matrix

The confusion matrix below shows per-class performance on the test set. Most classes are classified with high accuracy, with some expected confusion between visually similar classes (e.g., cats vs. dogs, automobiles vs. trucks).

![Confusion Matrix](Ekran%20Resmi%202026-08-17%2018.18.06.png)

## 🛠️ Tech Stack

- Python
- PyTorch / torchvision
- torchinfo
- scikit-learn (confusion matrix)
- seaborn & matplotlib (visualization)

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Abdussamedx/PaperReviewWorkNotebook.git
   cd PaperReviewWorkNotebook
   ```
2. Install the dependencies:
   ```bash
   pip install torch torchvision torchinfo scikit-learn seaborn matplotlib
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook PaperReviewWorkNotebook1.ipynb
   ```
   The CIFAR-10 dataset will be downloaded automatically on first run.

## 📌 Notes

This project was built primarily as a **learning exercise in reading and reimplementing a research paper from scratch**, focusing on understanding the internal mechanics of window-based attention and hierarchical vision transformers, rather than achieving state-of-the-art accuracy. As a result, hyperparameters and training length were kept modest for a full paper reproduction.

## 📬 Contact

Feel free to open an issue or reach out if you have questions or suggestions about the implementation.
