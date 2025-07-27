import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(config):
    """
    Creates and returns the data generators for training, validation, and testing.

    Args:
        config (dict): A dictionary containing data and training parameters.

    Returns:
        A tuple of (train_generator, val_generator, test_generator).
    """
    train_dir = os.path.join(config['data_dir'], 'train')
    val_dir = os.path.join(config['data_dir'], 'val')
    test_dir = os.path.join(config['data_dir'], 'test')
    img_size = tuple(config['image_size'])
    batch_size = config['batch_size']

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale validation and test data
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )

    print("✅ Data generators created successfully.")
    return train_generator, val_generator, test_generator