# Neural Network Compression with Quantisation and Pruning

[![arXiv](https://img.shields.io/badge/arXiv-2001.04850-b31b1b.svg)](https://arxiv.org/abs/2001.04850)

This repository reproduces and extends the work from the paper  
**"Quantisation and Pruning for Neural Network Compression and Regularisation"**  
by Kimessha Paupamah, Steven James, and Richard Klein.

Our implementation supports:
- **Iterative pruning** based on sensitivity scans
- **Per-channel quantisation**
- Overfitting correction through sparsification
- Multiple architectures: `AlexNet`, `MobileNetV2`, `ShuffleNetV2`
- Datasets: `CIFAR-10`, `FashionMNIST`

---

## 🧠 Background

This project explores how **pruning** and **quantisation** can be combined to compress deep neural networks - reducing storage and compute cost, and potentially improving generalisation.

We reproduce and analyse the findings of the original paper using **PyTorch 1.8+**, correcting bugs, updating APIs, and conducting additional experiments on modern CPUs/GPUs.

Read our full report [here](./Quantisation%20and%20Pruning%20for%20Neural%20Network%20Compression%20and%20Regularisation%20-%20Paper%20Reproduction.pdf).

---

## 🏗️ Methods

### 🔪 Pruning

We use two types of vanilla pruning:
- **Element-wise pruning**: Remove individual weights based on absolute or custom importance
- **Filter-wise pruning**: Remove entire filters based on norm metrics

The `Pruner` class in [`pruner.py`](./pruner.py) supports stage-wise pruning with configurable rules via `.rule` files.

### 🧮 Quantisation

We apply **per-channel quantisation** using PyTorch's built-in quantisation pipeline, adapted in [`quantiser.py`](./quantiser.py), including fusion and calibration steps for:
- AlexNet
- MobileNetV2
- ShuffleNetV2

Quantisation reduces model precision to 8-bit integers (int8), accelerating inference and shrinking file size.

---

## 🏃 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

```bash
python run.py --train 1 --arch alexnet --data fmnist
```

### 3. Prune a Model

```bash
python run.py --prune 1 --load AlexNet.pth --sensitivity 0.5
```

### 4. Quantise a Model
```bash
python run.py --quantise 1 --load AlexNet_pruned.pth
```

### 5. Overfit the Prune
```bash
python run.py --overfit 1 --load AlexNet.pth
python run.py --prune 1 --load AlexNet_overfitted.pth
```

## 📁 Project Structure
```bash
├── run.py                   # Main entrypoint
├── pruner.py                # Pruning logic
├── quantiser.py             # Quantisation logic
├── trainer.py               # Training and evaluation loop
├── utils.py                 # Model, dataset, and rule utilities
├── requirements.txt         # Dependencies
├── rules/                   # Pruning rule files
└── models/                  # Saved and quantised models
```

## 📊 Results Overview
| Architecture | Dataset      | Accuracy ↑ | Parameters ↓ | Size ↓        | Inference ↓  |
| ------------ | ------------ | ---------- | ------------ | ------------- | ------------ |
| AlexNet      | FashionMNIST | +1.13%     | 57M → 5M     | 217MB → 55MB  | 11ms → 5ms   |
| MobileNetV2  | CIFAR-10     | ≈0%        | 2.2M → 671K  | 8.7MB → 2.5MB | 35ms → 4.5ms |
| ShuffleNetV2 | FashionMNIST | ≈0%        | 1.2M → 815K  | 4.9MB → 1.4MB | 13ms → 7.5ms |

**Pruning** helps reduce overfitting and model size.
**Quantisation** provides speedups and compression with minimal accuracy loss.

## 🧪 Reproducing the Paper
This repo is a reproduction of the methods proposed in
📄 [Quantisation and Pruning for Neural Network Compression and Regularisation](https://www.arxiv.org/pdf/2001.04850)

Original code (broken): [kpaupamah/compression-and-regularisation](https://github.com/kpaupamah/compression-and-regularisation)

## ✍️ Authors
Yael & Yuval Dahari.  
BIU Department of Computer Science.

