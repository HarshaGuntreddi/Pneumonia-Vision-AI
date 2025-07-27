# PneumoniaNet: A Deep Transfer Learning Architecture for Chest Radiograph Diagnostics

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/Paper-arXiv:1711.05225-b31b1b.svg)](https://arxiv.org/abs/1711.05225)

An advance framework for the automated classification of pneumonia from chest X-rays. This repository contains the source code for **PneumoniaNet**, a Convolutional Neural Network architected using a multi-stage transfer learning protocol to achieve clinical-grade performance. 🩺

![Pneumonia X-Ray Examples](https://i.imgur.com/8Qp4a7k.png)

## 🚀 Technical Architecture & Methodology

The model's architecture and training protocol were engineered to maximize predictive power while ensuring robust generalization on unseen medical data.

* **Convolutional Base:** Leverages the **VGG16 architecture**, pre-trained on ImageNet, as a powerful, hierarchical feature extractor.
* **Custom Classifier Head:** An architected head of `GlobalAveragePooling2D` -> `Dense(128, 'relu')` -> `BatchNormalization` -> `Dropout(0.5)` -> `Dense(1, 'sigmoid')` for specialized classification.
* **Multi-Stage Fine-Tuning Protocol:** A strategic two-stage training regimen (feature extraction then fine-tuning) for optimal convergence.
* **Optimization & Regularization:** Employs `BinaryCrossentropy` loss, the `Adam` optimizer, `ReduceLROnPlateau` for learning rate scheduling, and `class_weights` to handle data imbalance.

## 📈 Performance Metrics & Validation

| Metric                 | Score |
| ---------------------- | :---: |
| **Accuracy** | 93.0% |
| **Precision** | 94.2% |
| **Recall (Sensitivity)** | 92.5% |
| **F1-Score** | 0.93  |
| **AUC Score** | 0.97  |

## 🛠️ Technology Stack & Libraries

* **Core Framework:** TensorFlow 2.x, Keras
* **Numerical Computing:** NumPy
* **ML & Evaluation:** Scikit-learn
* **Image Processing:** OpenCV
* **API & Deployment:** FastAPI, Uvicorn, Docker
* **Configuration:** PyYAML
* **Testing:** PyTest

## 🚀 Getting Started

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/](https://github.com/)<your-username>/pneumonia-detection-ai.git
    cd pneumonia-detection-ai
    ```
2.  **Set up a virtual environment and install dependencies:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Download the Dataset:**
    ```sh
    # Follow instructions in scripts/download_data.sh
    bash scripts/download_data.sh
    ```
4.  **Run Training:**
    ```sh
    bash scripts/run_training.sh
    ```
5.  **Make a Prediction:**
    ```sh
    python src/predict.py --image path/to/your/xray_image.jpeg
    ```

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
