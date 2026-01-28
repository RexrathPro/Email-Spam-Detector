from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from preprocess import transform_text

app = FastAPI()

# Load artifacts
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

class Email(BaseModel):
    message: str

@app.post("/predict")
def predict_spam(email: Email):
    transformed = transform_text(email.message)
    vector = vectorizer.transform([transformed])
    prediction = model.predict(vector)[0]
    return {
        "prediction": "spam" if prediction == 1 else "ham",
        "label": int(prediction)
    }
