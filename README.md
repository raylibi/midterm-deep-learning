# Deep Learning Midterm Exam Repository

## Student Identification
* **Name:** Rayhan Diff
* **NIM:** 1103220039
* **Class:** [Insert Your Class Here]

---

## Repository Purpose
This repository serves as a submission for the Deep Learning Midterm Exam. It contains two distinct projects demonstrating the application of Deep Neural Networks (DNN) for different machine learning tasks:
1.  **Binary Classification:** Detecting fraudulent transactions.
2.  **Regression:** Predicting the release year of songs based on audio features.

---

## Project Overviews & Model Descriptions

### 1. Online Transaction Fraud Detection
* **File:** MidtermDL_1.ipynb
* **Task:** Binary Classification (Fraud vs. Normal)
* **Framework:** TensorFlow / Keras

#### Project Overview
This project aims to identify fraudulent online transactions from a dataset containing transaction details and identity features. The dataset is highly imbalanced, requiring specific handling techniques like class weighting to ensure the model learns to detect the minority class (fraud) effectively.

#### Model Architecture
A Multi-Layer Perceptron (MLP) model was designed to handle the 394 input features:
* **Input Layer:** 394 features.
* **Hidden Layers:**
    * Dense (512 neurons, ReLU activation) + BatchNormalization + Dropout (0.3)
    * Dense (256 neurons, ReLU activation) + BatchNormalization + Dropout (0.3)
* **Output Layer:** Dense (1 neuron, Sigmoid activation) for probability output.
* **Optimizer:** Adam (Learning rate: 1e-3).
* **Loss Function:** Binary Crossentropy.

#### Matrix Results & Performance
* **Validation AUC Score:** 0.9353 (Epoch 30).
* **Training AUC:** ~0.95.
* **Conclusion:** The model demonstrates strong discriminative ability between fraud and normal transactions using Weighted Loss to handle class imbalance.

---

### 2. Song Year Prediction
* **File:** midtermdl_2.ipynb
* **Task:** Regression
* **Framework:** PyTorch

#### Project Overview
This project focuses on predicting the release year of a song based on 90 diverse audio timbre features (derived from the YearPredictionMSD dataset). The goal is to minimize the difference between the predicted year and the actual release year.

#### Model Architecture
A Deep Neural Network implemented in PyTorch:
* **Input Layer:** 90 audio features.
* **Hidden Layers:**
    * Linear (128 neurons) + ReLU + Dropout (0.2)
    * Linear (64 neurons) + ReLU + Dropout (0.2)
    * Linear (32 neurons) + ReLU
* **Output Layer:** Linear (1 neuron) for year prediction.
* **Optimizer:** Adam.
* **Loss Function:** Mean Squared Error (MSE).

#### Matrix Results & Performance
The model was evaluated on a 20% test split:
* **MSE (Mean Squared Error):** 74.60.
* **RMSE (Root Mean Squared Error):** 8.64 (The model misses the actual year by approximately 8.64 years on average).
* **MAE (Mean Absolute Error):** 5.99.
* **R-squared Score:** 0.3732 (Explains approximately 37% of the variance in the data).

---

## How to Navigate Through the Notebooks

1.  **Environment:** These notebooks are designed to run in Google Colab or a local Jupyter environment with GPU support (NVIDIA CUDA).
2.  **Data Requirement:**
    * The notebooks attempt to mount Google Drive to load datasets.
    * Ensure the datasets (`train_transaction.csv`, `test_transaction.csv`, and `midterm-regresi-dataset.csv`) are placed in the path: `/content/drive/MyDrive/Dataset_MLDL/`.
    * Alternatively, you can modify the `base_path` variable in the code to point to your local data directory.
3.  **Running the Code:**
    * Open `MidtermDL_1.ipynb` for the Classification task (TensorFlow).
    * Open `midtermdl_2.ipynb` for the Regression task (PyTorch).
    * Run cells sequentially (from top to bottom).

## Libraries Used
* **Python 3**
* **TensorFlow / Keras** (for Project 1)
* **PyTorch** (for Project 2)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Preprocessing & Metrics)
* **Matplotlib & Seaborn** (Visualization)