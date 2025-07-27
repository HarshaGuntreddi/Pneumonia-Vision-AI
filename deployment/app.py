import numpy as np
import cv2
import yaml
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import io

app = FastAPI(title="Pneumonia Detection API")

# Load config and model at startup
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
model = load_model(config['model_path'])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the image
    img_size = tuple(config['image_size'])
    img_resized = cv2.resize(img, img_size)
    img_rescaled = img_resized / 255.0
    img_expanded = np.expand_dims(img_rescaled, axis=0)

    # Make prediction
    prediction = model.predict(img_expanded)[0][0]
    label_index = 1 if prediction > 0.5 else 0
    label = config['class_names'][label_index]
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return {"prediction": label, "confidence": confidence}