# Data Pipeline

The data pipeline is designed to efficiently load, preprocess, and augment chest X-ray images for training and evaluation.

## 1. Data Source

* **Dataset**: [Chest X-Ray Images (Pneumonia) from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
* **Structure**: The data is pre-organized into `train`, `val`, and `test` directories, each containing `NORMAL` and `PNEUMONIA` subdirectories.

## 2. Data Loading

* **Tool**: We use the `ImageDataGenerator` class from `tensorflow.keras.preprocessing.image`.
* **Process**: It automatically loads images from the directories, infers class labels from the folder names, and creates batches of data.

## 3. Preprocessing

All images, regardless of the dataset split (train, val, or test), undergo the following preprocessing steps:
1.  **Resizing**: Images are resized to `(224, 224)` pixels to match the input size required by the VGG16 model.
2.  **Rescaling**: Pixel values are normalized from the range `[0, 255]` to `[0, 1]` by dividing by 255. This is a standard practice that helps the model converge faster.

## 4. Data Augmentation (Training Set Only)

To prevent overfitting and improve the model's ability to generalize, we apply on-the-fly data augmentation to the training images. This synthetically expands the dataset by creating modified versions of the images in each batch.

The augmentations include:
* `rotation_range=20`: Randomly rotate images by up to 20 degrees.
* `width_shift_range=0.2`: Randomly shift images horizontally.
* `height_shift_range=0.2`: Randomly shift images vertically.
* `shear_range=0.2`: Apply shear transformations.
* `zoom_range=0.2`: Randomly zoom into images.
* `horizontal_flip=True`: Randomly flip images horizontally.
* `fill_mode='nearest'`: Fill any new pixels created by transformations with the nearest available pixel value.

**Note**: Augmentation is **only** applied to the training set. The validation and test sets are left untouched (except for rescaling) to ensure that we are evaluating the model's performance on unmodified, real-world-like data.