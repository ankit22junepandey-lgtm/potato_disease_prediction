from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Potato Disease Detection API", version="1.0.0")

# Load your TensorFlow/Keras model
model = load_model("model.h5")

# Define your class labels (change according to your dataset)
class_labels = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
def home():
    return {"message": "Welcome to Potato Disease Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((256, 256))  # same as image_size used in training
        img_array = np.array(img) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})
