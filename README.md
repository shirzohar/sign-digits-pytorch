# 🧠 Sign Language Digits Classification (PyTorch)

## 📌 Overview
This project implements a deep learning solution using **fully connected neural networks (FCNN)** in **PyTorch** to classify hand signs representing digits 0–9, based on the **Sign Language Digits Dataset**. The project was done as part of a mid-semester milestone in the "Basics of Deep Learning" course.

The project includes:
- ✅ **Binary classification** – distinguish between two specific digits ("4" vs. "5").
- ✅ **Multi-class classification** – recognize all ten digits (0–9).
- ✅ Architecture & hyperparameter experiments.

---

## 📂 Files
```
sign-language-classification/
├── basics_2025.ipynb   # Main Colab notebook
├── report.pdf          # Final project report
├── README.md                               # Project overview and usage
├── requirements.txt                        # Python dependencies
```

---

## 📊 Dataset
- **Name:** Sign Language Digits Dataset
- **Size:** 5,000 grayscale images (28x28 pixels)
- **Classes:** Digits 0–9
- **Preprocessing:**
  - Normalized pixel values to [0, 1]
  - Flattened images to 784-dim vectors
  - Relabeled subset for binary classification (4 → 0, 5 → 1)
  - Split into train (80%), val (10%), test (10%)

---

## 🏗️ Model Architectures
### 🔹 Binary Classification Model
- Input: 784 neurons
- Hidden layers: [128 → 64], activation: ReLU
- Output: 1 neuron, activation: Sigmoid
- Loss: `BCELoss`
- Accuracy: **92.8%** (Test)

### 🔹 Base Model (Multi-class)
- Input: 784 neurons
- Hidden layers: [128 → 64], ReLU + Dropout(0.2)
- Output: 10 neurons, activation: Log-Softmax
- Loss: `CrossEntropyLoss`
- Accuracy: **45.4%** (Test)

### 🔹 Experiment 1: Deeper Network
- Hidden layers: [512 → 256 → 128], ReLU
- Accuracy: **97.1%** (Test)

### 🔹 Experiment 2: Hyperparameter Tuning
- Optimizer: SGD
- LR: 0.01, Batch Size: 64, Dropout: 0.3
- Accuracy: **99.0%** (Test)

---

## 📈 Evaluation Metrics
- Accuracy (Train, Validation, Test)
- Confusion Matrix
- Classification Report (Precision, Recall, F1)
- Training & Validation Loss / Accuracy graphs

---

## 🚀 Run Instructions
1. Clone repository or open the notebook in **Google Colab**
2. Ensure dataset `.npy` files are available in your environment
3. Run notebook cells sequentially:
   - Data loading & preprocessing
   - Model training
   - Evaluation & visualization

---

## 🏆 Final Results
| Model              | Test Accuracy | Train Acc | Val Acc |
|-------------------|---------------|-----------|---------|
| Base Model        | 45.4%         | 45.7%     | 45.9%   |
| Experiment 1      | 97.1%         | 98.9%     | 97.0%   |
| Experiment 2 ⭐    | **99.0%**     | 99.5%     | 98.8%   |

---

## 👥 Authors
- **Shir Zohar**  
- **Sivan Zagdon**

---

## 📜 License
This project is for academic use as part of the Deep Learning course @ Colman College.
