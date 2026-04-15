from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import sys

app = FastAPI(title="Spam Detector API")

# Setup CORS to allow the Vite React Dev Server to fetch data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamically resolve absolute paths to the models directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from src.preprocess import transform_text

model_path = os.path.join(base_dir, "models", "model.pkl")
vectorizer_path = os.path.join(base_dir, "models", "vectorizer.pkl")

# Pre-load the models in memory
try:
    print("Loading vectorizer and model...")
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading models. Make sure they exist at {model_path} and {vectorizer_path}\nError: {e}")
    model = None
    vectorizer = None

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_email(request: EmailRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    
    # 1. Preprocess and vectorizer text features
    transformed_text = transform_text(request.text)
    vect_text = vectorizer.transform([transformed_text])
    
    # 2. Extract Prediction
    prediction = model.predict(vect_text)[0]
    
    # Predict normally returns 0 (Ham) or 1 (Spam), though it depends on your specific training strings.
    # Map accordingly to "spam" or "ham"
    pred_str = str(prediction).lower()
    is_spam = pred_str in ['1', 'spam', 'true']
    
    # 3. Extract Prediction Probability for Confidence
    confidence = 100.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vect_text)[0]
        confidence = max(proba) * 100

    return {
        "type": "spam" if is_spam else "ham",
        "confidence": round(float(confidence), 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
