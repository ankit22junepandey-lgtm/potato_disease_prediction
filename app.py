from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the saved models + vectorizer
bundle = joblib.load("news_models.pkl")
vectorizer = bundle["vectorizer"]
LR = bundle["LR"]
DT = bundle["DT"]
GB = bundle["GB"]
RF = bundle["RF"]

# FastAPI app
app = FastAPI(title="Fake News Detection API")

# Input schema
class NewsInput(BaseModel):
    text: str

# Output label function
def output_label(n):
    return "✅ Real News" if n == 1 else "❌ Fake News"

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Fake News Detection API! Use /predict to test."}

# Prediction endpoint
@app.post("/predict/")
def predict(news: NewsInput):
    # Vectorize input text
    new_xv_test = vectorizer.transform([news.text])

    # Predictions
    pred_LR = LR.predict(new_xv_test)[0]
    pred_DT = DT.predict(new_xv_test)[0]
    pred_GB = GB.predict(new_xv_test)[0]
    pred_RF = RF.predict(new_xv_test)[0]

    return {
        "Logistic Regression": output_label(pred_LR),
        "Decision Tree": output_label(pred_DT),
        "Gradient Boosting": output_label(pred_GB),
        "Random Forest": output_label(pred_RF)
    }
