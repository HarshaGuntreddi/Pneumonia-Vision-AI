import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from data_loader import get_data_generators
from model import build_model
from utils import plot_training_history, evaluate_model

def train():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])
    
    os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)

    train_gen, val_gen, test_gen = get_data_generators(config)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Calculated Class Weights: {class_weights}")

    model = build_model(input_shape=(*config['image_size'], 3))

    checkpoint = ModelCheckpoint(config['model_path'], monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    print("\n--- STAGE 1: TRAINING CLASSIFIER HEAD ---")
    model.get_layer('vgg16').trainable = False
    model.compile(optimizer=Adam(learning_rate=config['learning_rate_stage_1']), loss='binary_crossentropy', metrics=['accuracy'])
    history1 = model.fit(
        train_gen, epochs=config['epochs_stage_1'], validation_data=val_gen,
        class_weight=class_weights, callbacks=[checkpoint, lr_reducer]
    )

    print("\n--- STAGE 2: FINE-TUNING ENTIRE MODEL ---")
    vgg_base = model.get_layer('vgg16')
    vgg_base.trainable = True
    for layer in vgg_base.layers[:-4]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=config['learning_rate_stage_2']), loss='binary_crossentropy', metrics=['accuracy'])
    history2 = model.fit(
        train_gen, epochs=config['epochs_stage_2'], validation_data=val_gen,
        class_weight=class_weights, callbacks=[checkpoint, lr_reducer, early_stopper]
    )

    print(f"\nLoading best model from {config['model_path']}")
    model.load_weights(config['model_path'])
    evaluate_model(model, test_gen, config)
    
    # Combine histories for plotting
    history1.history['accuracy'].extend(history2.history['accuracy'])
    history1.history['val_accuracy'].extend(history2.history['val_accuracy'])
    history1.history['loss'].extend(history2.history['loss'])
    history1.history['val_loss'].extend(history2.history['val_loss'])
    plot_training_history(history1, config)

if __name__ == '__main__':
    train()
