# MNIST Digit Classification using Neural Networks

## 📌 Project Overview
This project implements a **Neural Network (NN)** model to classify handwritten digits (0–9) from the **MNIST dataset**. The MNIST dataset consists of **28×28 grayscale images** of handwritten digits, widely used for benchmarking classification algorithms in computer vision.

The notebook trains a neural network from scratch using deep learning techniques to achieve high accuracy on both training and test sets.

---

## 📂 Dataset
- **Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- **Training Data**: 60,000 images
- **Testing Data**: 10,000 images
- **Image Size**: 28 × 28 pixels (flattened to 784 features)
- **Labels**: Digits 0 to 9 (10 classes)

---

## ⚙️ Project Workflow
1. **Import Libraries** – NumPy, Matplotlib, TensorFlow/Keras, etc.
2. **Load Dataset** – MNIST data from Keras datasets API.
3. **Data Preprocessing**
   - Normalize pixel values to range [0, 1]
   - Flatten images for NN input
   - Convert labels to one-hot encoded vectors
4. **Build Neural Network**
   - Input Layer: 784 neurons (28×28 pixels)
   - Hidden Layers: Dense layers with activation functions (ReLU)
   - Output Layer: 10 neurons (Softmax activation)
5. **Compile Model**
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Metric: Accuracy
6. **Train Model**
   - Epochs: configurable
   - Batch size: configurable
7. **Evaluate Model**
   - Accuracy on test dataset
   - Confusion matrix and classification report
8. **Visualize Results**
   - Training vs validation accuracy/loss curves
   - Sample predictions

---

---

## 📊 Results
- **Test Accuracy**: ~97–98% (depending on hyperparameters)
- High performance achieved with simple architecture due to MNIST’s simplicity.
- Misclassification analysis shows most errors occur with visually similar digits (e.g., 4 vs 9, 5 vs 3).

---

## 🚀 How to Run
1. Clone this repository or download the notebook.
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib

jupyter notebook DL_Project_2_MNIST_Digit_classification_using_NN.ipynb


