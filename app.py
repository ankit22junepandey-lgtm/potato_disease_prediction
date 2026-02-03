from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from keras.layers import TFSMLayer
from keras.models import Sequential

app = FastAPI()

# ✅ Load SavedModel using TFSMLayer (Keras 3 compatible)
model = Sequential([
    TFSMLayer(
        "my_plant_disease_model/my_plant_disease_model",
        call_endpoint="serve"
    )
])

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

IMAGE_SIZE = 256

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess(image_bytes)

    predictions = model(img)
    predictions = predictions.numpy()

    index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "prediction": CLASS_NAMES[index],
        "confidence": round(confidence * 100, 2)
    }
