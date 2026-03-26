import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load model and vectorizer

import os
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/best_model.pkl'))
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
vectorizer = model_data['vectorizer']

def preprocess_text(text):
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


def get_risk_score(label):
    # Map string label to risk score (percent)
    mapping = {
        'minimum': 10,
        'mild': 30,
        'moderate': 60,
        'severe': 90,
        'extreme': 100
    }
    return mapping.get(label.lower(), 0)

def get_coping_strategies(label):
    # Always return at least 4 strategies
    base = [
        "Maintain social connections",
        "Exercise regularly",
        "Talk to friends",
        "Practice mindfulness",
        "Seek professional help",
        "Join support groups",
        "Immediate professional intervention",
        "Contact helplines",
        "Self-care",
        "Reach out for help"
    ]
    # Pick 4 unique strategies based on severity
    label = label.lower()
    if label == 'minimum':
        return base[:4]
    elif label == 'mild':
        return base[2:6]
    elif label == 'moderate':
        return base[4:8]
    elif label == 'severe' or label == 'extreme':
        return base[6:10]
    else:
        return base[:4]

if __name__ == "__main__":
    print("Enter the text to predict depression severity (type 'stop' to exit):")
    while True:
        user_input = input()
        if user_input.strip().lower() == 'stop':
            print("Exiting prediction loop.")
            break
        if not user_input.strip():
            print("No input provided. Please enter text or type 'stop' to exit.")
            continue
        clean_text = preprocess_text(user_input)
        features = vectorizer.transform([clean_text])
        label = model.predict(features)[0]
        risk_score = get_risk_score(label)
        coping_strategies = get_coping_strategies(label)
        print(f"Depression Level: {label}")
        print(f"Risk Score: {risk_score}%")
        print("Coping Strategies:")
        for idx, strategy in enumerate(coping_strategies, 1):
            print(f"  {idx}. {strategy}")
        if label.lower() in ["severe", "extreme"] or "suicide" in user_input.lower() or "kill myself" in user_input.lower():
            print("ALERT: Possible suicide risk detected! Immediate attention required.")
        print("\n---\nEnter another text or type 'stop' to exit:")
