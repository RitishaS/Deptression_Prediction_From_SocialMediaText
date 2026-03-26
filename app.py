"""
Flask application for Depression Level Analyzer
Integrates with the ML backend for depression prediction
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import sys
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ================================
# LOAD PREPROCESSED DATA & MODEL
# ================================

# Get paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'src', 'preprocessed_data.pkl')

# Load model
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data.get('model')
    app.logger.info("✅ Model loaded successfully")
except FileNotFoundError:
    app.logger.error("❌ Model file not found!")
    model = None
except Exception as e:
    app.logger.error(f"❌ Error loading model: {e}")
    model = None

# Load preprocessor/vectorizer
try:
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor_data = pickle.load(f)
    vectorizer = preprocessor_data.get('vectorizer')
    app.logger.info("✅ Vectorizer loaded successfully")
except Exception as e:
    app.logger.warning(f"⚠️ Vectorizer not found, will use direct preprocessing")
    vectorizer = None

# Initialize NLP components
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# Slang dictionary
SLANG_DICT = {
    "lol": "laugh", "lmao": "laugh", "rofl": "laugh", "haha": "laugh",
    "omg": "oh my god", "wtf": "what the", "smh": "shaking my head",
    "tbh": "to be honest", "idk": "i do not know", "nvm": "never mind",
    "imo": "in my opinion", "btw": "by the way", "asap": "as soon as possible",
    "fyi": "for your information", "u": "you", "ur": "your", "r": "are",
    "b": "be", "c": "see", "d": "the", "m": "am", "n": "and",
    "ppl": "people", "thru": "through", "2": "to", "4": "for",
    "w8": "wait", "gr8": "great", "b4": "before", "2day": "today",
    "2nite": "tonight", "ne1": "anyone", "sum1": "someone",
}

# Regex patterns
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
EMAIL_PATTERN = re.compile(r'\S+@\S+')
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE
)

# ================================
# PREPROCESSING FUNCTION
# ================================

def preprocess_text(text):
    """
    Preprocess text following the same pipeline as training
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = URL_PATTERN.sub('', text)
    
    # Remove email addresses
    text = EMAIL_PATTERN.sub('', text)
    
    # Remove emojis
    text = EMOJI_PATTERN.sub('', text)
    
    # Replace slangs
    words = text.split()
    words = [SLANG_DICT.get(word, word) for word in words]
    text = ' '.join(words)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]
    
    # Lemmatization
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# ================================
# HELPER FUNCTIONS
# ================================

def get_severity_level(prediction_value):
    """Map prediction to severity level"""
    if prediction_value >= 0.7:
        return "Severe"
    elif prediction_value >= 0.5:
        return "Moderate"
    elif prediction_value >= 0.3:
        return "Mild"
    else:
        return "None"

def get_coping_strategies(severity):
    """Get personalized coping strategies based on severity"""
    strategies_map = {
        "None": [
            "Continue maintaining regular sleep schedule",
            "Stay physically active with exercise you enjoy",
            "Nurture your relationships and social connections",
            "Practice mindfulness or meditation daily",
            "Engage in hobbies and activities you love"
        ],
        "Mild": [
            "Talk to someone you trust about your feelings",
            "Practice regular exercise (at least 30 minutes daily)",
            "Maintain a healthy sleep routine",
            "Try journaling to express your emotions",
            "Consider speaking with a therapist or counselor"
        ],
        "Moderate": [
            "Seek professional mental health support urgently",
            "Reach out to trusted friends or family members",
            "Practice self-care activities daily",
            "Avoid isolating yourself; stay connected",
            "Consider professional medication consultation if needed"
        ],
        "Severe": [
            "Contact a mental health crisis hotline immediately",
            "Reach out to a trusted mental health professional",
            "Call 988 (Suicide & Crisis Lifeline) if in the US",
            "Tell someone close to you how you're feeling",
            "Avoid alcohol and limit caffeine intake"
        ]
    }
    
    return strategies_map.get(severity, strategies_map["Mild"])

# ================================
# ROUTES
# ================================

@app.route('/')
def home():
    """Serve the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict depression severity from user text
    """
    try:
        # Get text from form
        user_text = request.form.get('text', '').strip()
        
        if not user_text:
            return jsonify({
                'error': 'No text provided',
                'prediction': 'No Depression 😊 (Confidence: 0.00)'
            }), 400
        
        if len(user_text) < 10:
            return jsonify({
                'error': 'Text too short',
                'prediction': 'No Depression 😊 (Confidence: 0.00)'
            }), 400
        
        # Preprocess text
        processed_text = preprocess_text(user_text)
        
        if not processed_text:
            return jsonify({
                'error': 'Text could not be processed',
                'prediction': 'No Depression 😊 (Confidence: 0.00)'
            }), 400
        
        # Vectorize text
        if vectorizer is None:
            return jsonify({
                'error': 'Vectorizer not available',
                'prediction': 'No Depression 😊 (Confidence: 0.00)'
            }), 500
        
        X = vectorizer.transform([processed_text])
        
        # Make prediction
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'prediction': 'No Depression 😊 (Confidence: 0.00)'
            }), 500
        
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        
        # Generate user-friendly message
        if prediction == 1:
            message = f"Depression Detected 😔 (Confidence: {prob:.2f})"
        else:
            message = f"No Depression 😊 (Confidence: {1-prob:.2f})"
        
        # Determine severity
        severity = get_severity_level(prob)
        
        # Get coping strategies
        strategies = get_coping_strategies(severity)
        
        # Determine suicide risk
        suicide_risk = "High" if prob >= 0.75 else "Moderate" if prob >= 0.5 else "Low"
        
        # Return response
        response = {
            'prediction': message,
            'severity': severity,
            'risk_score': round(prob * 100, 1),
            'suicide_risk': suicide_risk,
            'emotion': 'Depression' if prediction == 1 else 'Wellness',
            'strategies': strategies,
            'success': True
        }
        
        app.logger.info(f"✅ Prediction made: {severity} (confidence: {prob:.2f})")
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"❌ Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'prediction': 'No Depression 😊 (Confidence: 0.00)',
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

# ================================
# ERROR HANDLERS
# ================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ================================
# RUN APP
# ================================

if __name__ == '__main__':
    if model is None or vectorizer is None:
        print("⚠️  Warning: Model or vectorizer not loaded. Please ensure 'best_model.pkl' and 'src/preprocessed_data.pkl' exist.")
    
    print("🚀 Starting Depression Level Analyzer...")
    print("📊 Visit http://localhost:5000 in your browser")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
