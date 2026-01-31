from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Potato Disease Detection API")

# Load model
MODEL_PATH = "model(2).h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Must match training order
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

IMAGE_SIZE = 256

@app.get("/")
def home():
    return {"message": "Potato Disease Detection API is running"}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        predictions = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
