from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Potato Disease Detection API")

# Load trained model
model = tf.keras.models.load_model("potato_model.h5")

# ⚠️ Change order ONLY if your training labels order was different
class_names = [
    "Healthy",
    "Early Blight",
    "Late Blight"
]

IMAGE_SIZE = 256

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0   # Same normalization as training
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "Potato Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        predictions = model.predict(img)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return {
            "predicted_disease": class_names[class_index],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
