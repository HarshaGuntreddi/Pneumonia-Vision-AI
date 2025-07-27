import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history, config):
    save_path = os.path.join(config['output_dir'], 'training_history.png')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs, acc, 'bo-', label='Training acc')
    ax1.plot(epochs, val_acc, 'ro-', label='Validation acc')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax2.plot(epochs, loss, 'bo-', label='Training loss')
    ax2.plot(epochs, val_loss, 'ro-', label='Validation loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()

    plt.savefig(save_path)
    print(f"✅ Training history plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, config):
    save_path = os.path.join(config['output_dir'], 'confusion_matrix.png')
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()

def evaluate_model(model, test_generator, config):
    print("\nEvaluating model on the test set...")
    y_pred_proba = model.predict(test_generator)
    y_pred = (y_pred_proba > 0.5).astype("int32").reshape(-1)
    
    y_true = test_generator.classes
    class_names = config['class_names']

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names, config)