import numpy as np
import cv2
import argparse
import os
import yaml
from tensorflow.keras.models import load_model

def predict_image(config, image_path):
    try:
        model_path = config['model_path']
        img_size = tuple(config['image_size'])
        class_names = config['class_names']

        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        img = cv2.resize(img, img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label_index = 1 if prediction > 0.5 else 0
        label = class_names[label_index]

        print("\n--- Prediction Result ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print("-------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()
    
    with open('configs/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    predict_image(cfg, args.image)