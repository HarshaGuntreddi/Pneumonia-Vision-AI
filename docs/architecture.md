# Model Architecture: Pneumonia-Vision-AI

The model is a Convolutional Neural Network (CNN) specifically designed for medical image classification. It employs a transfer learning strategy to leverage knowledge from a large-scale dataset (ImageNet) and fine-tune it for the specific task of pneumonia detection.

## 1. Convolutional Base: VGG16

* **Model**: We use the **VGG16** architecture as our convolutional base.
* **Pre-training**: The weights are pre-trained on the ImageNet dataset. This is crucial as the initial layers of VGG16 are adept at recognizing generic features like edges, textures, and patterns, which are also present in medical radiographs.
* **Excluding Top Layers**: The original VGG16's fully connected (top) layers, which are specific to its original 1000-class classification task, are removed (`include_top=False`).

## 2. Custom Classifier Head

A new classification head is added on top of the VGG16 base to adapt the model for our binary (Pneumonia vs. Normal) task.

The data flows through the following layers:
1.  **Global Average Pooling 2D (`GlobalAveragePooling2D`)**: This layer dramatically reduces the number of parameters by taking the average of each feature map from the VGG16 base. It converts a 3D tensor into a 1D feature vector, making the model less prone to overfitting.
2.  **Fully Connected Layer (`Dense`)**: A `Dense` layer with 128 neurons and a **ReLU** activation function is used to learn non-linear combinations of the extracted features.
3.  **Batch Normalization (`BatchNormalization`)**: Applied after the `Dense` layer to normalize the activations. This stabilizes and accelerates the training process.
4.  **Dropout (`Dropout`)**: A `Dropout` layer with a rate of 0.5 is used for regularization. It randomly sets 50% of the input units to 0 during training, preventing the model from becoming too reliant on any single neuron.
5.  **Output Layer (`Dense`)**: The final layer is a `Dense` layer with a single neuron and a **Sigmoid** activation function. It outputs a value between 0 and 1, representing the probability of the input image being classified as 'Pneumonia'.