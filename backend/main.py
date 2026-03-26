
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import os
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Download required NLTK data (if not already present)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the trained model and vectorizer
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/best_model.pkl'))
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
vectorizer = model_data['vectorizer']

class TextInput(BaseModel):
    text: str


def preprocess_text(text):
    """
    Clean and preprocess text data (same as training)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def get_depression_label(pred):
    # Map model output to label (customize as per your model)
    labels = ['Mild', 'Moderate', 'Severe', 'Extreme']
    try:
        return labels[int(pred)]
    except:
        return 'Unknown'

def get_risk_score(pred):
    # Dummy risk score logic (customize as needed)
    return float(0.25 + 0.25 * int(pred))

def get_coping_strategies(label):
    # Dummy strategies (customize as needed)
    strategies = {
        'Mild': ["Maintain social connections", "Exercise regularly"],
        'Moderate': ["Talk to friends", "Practice mindfulness"],
        'Severe': ["Seek professional help", "Join support groups"],
        'Extreme': ["Immediate professional intervention", "Contact helplines"]
    }
    return strategies.get(label, ["Self-care", "Reach out for help"])

@app.post("/predict")
def predict_depression(input: TextInput):
    # Preprocess input text
    clean_text = preprocess_text(input.text)
    features = vectorizer.transform([clean_text])
    # Predict using the loaded model
    pred = model.predict(features)[0]
    label = get_depression_label(pred)
    risk_score = get_risk_score(pred)
    coping_strategies = get_coping_strategies(label)
    return {
        "depression_level": label,
        "risk_score": risk_score,
        "coping_strategies": coping_strategies
    }

@app.get("/")
def root():
    return {"message": "Depression Prediction API is running."}
