import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np

app = FastAPI()

MODEL_PATH = os.path.join(os.getcwd(), "model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

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
    image = Image.open(file.file).convert("RGB")
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return {
        "prediction": predicted_class,
        "confidence": confidence
    }
