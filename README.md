# Deep Learning for Fashion MNIST: A Comparative Study (2025)

> **Author:** Muhammad Mubashar Shahzad 
> **University:** University of Trieste 
> **Date:** 07/2025

---

## 📚 Project Overview

This project explores and compares multiple machine learning and deep learning models for the Fashion MNIST image classification benchmark. The goal is to demonstrate the performance improvements from classical machine learning (Logistic Regression), through fully connected (dense) neural networks, to advanced Convolutional Neural Networks (CNNs) with automated hyperparameter tuning.

---

## 🗂️ Dataset

- **Fashion MNIST:** 70,000 grayscale images (28x28 pixels) of clothing items, split into 60,000 training and 10,000 test images.
- **Classes:** 10 categories (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Download Source:** [Keras Datasets API](https://keras.io/api/datasets/fashion_mnist/) (loaded automatically in notebook).

---

## ⚙️ Workflow

1. **Data Loading & Preprocessing**
    - Normalize pixel values to [0, 1].
    - Prepare images as flattened vectors (for classical ML & dense NN) and as 2D arrays with channel (for CNN).
    - One-hot encode labels for neural networks.
    - Visualize class distribution and sample images.

2. **Model 1: Logistic Regression (Baseline)**
    - Treats each image as a flat vector.
    - Provides a reference for advanced models.

3. **Model 2: Dense Neural Network**
    - Two hidden layers with ReLU activation.
    - Models non-linear relationships, but does not leverage image spatial structure.

4. **Model 3: Convolutional Neural Network (CNN)**
    - Two convolutional layers (with max pooling), followed by dense and dropout layers.
    - Learns spatial features (edges, textures, patterns) automatically.
    - Yields significant performance improvement.

5. **Hyperparameter Tuning (Keras Tuner)**
    - Automated random search for optimal Conv2D filters, kernel sizes, dense units, and dropout rate.
    - Trains the best-found CNN for final evaluation.

6. **Evaluation & Comparison**
    - Metrics: Test accuracy, confusion matrix, precision, recall, F1-score.
    - Visualizations: Bar plots, grouped classification reports, comparison tables.

---

## 🏆 Results

| Model                | Test Accuracy | Remarks                                     |
|----------------------|--------------|---------------------------------------------|
| Logistic Regression  | 0.8438%     | Baseline, no spatial feature learning       |
| Dense Neural Network | 0.8822%     | Nonlinear modeling, improved over baseline  |
| CNN                  | 0.9121%     | Best default model, learns spatial patterns |
| Tuned CNN            | 0.9197%     | Highest accuracy after hyperparameter search|


- **Best model:** Tuned CNN, with validation accuracy of 0.9197%.
- **Classification metrics and confusion matrices** available in the notebook outputs.

---

## 📈 Visualizations

- Sample images from all classes.
- Confusion matrices for each model.
- Bar plots of test accuracy and classification metrics.
- Hyperparameter search space and best parameters found.

---

## 🚀 How to Run

### **Google Colab**
1. Open the main notebook (`Fashion_MNIST_CNN_Project_2025.ipynb`) in Google Colab.
2. Run all cells in sequence.
3. If prompted to install Keras Tuner, run the install cell:
    ```python
    !pip install keras-tuner --quiet
    ```
4. All results, figures, and comparison tables will be generated in-place.

### **Locally**
1. Clone this repository:
    ```
    git clone https://github.com/mubasharshahzad64/Deep-learning-Project_2025.git
    cd deep-learning-project_2025
    ```
2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook in Jupyter or JupyterLab.

---

## 🧰 Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- keras / tensorflow
- keras-tuner

Install requirements via:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras keras-tuner
