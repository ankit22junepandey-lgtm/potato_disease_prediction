from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Potato Disease Detection API")

# Load model
model = load_model("model.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "Potato Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed = preprocess_image(image)

    predictions = model.predict(processed)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }
