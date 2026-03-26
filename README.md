# Depression Level Analyzer

A beautiful, modern web application for depression severity prediction from social media text using Machine Learning.

## 📁 Project Structure

```
SocialMedia_Text_Depression/
├── frontend/                      # Frontend files (HTML, CSS, JS)
│   ├── index.html                # Main application page
│   ├── styles.css                # Beautiful styling
│   └── script.js                 # Frontend logic & API calls
│
├── backend/                       # Backend API server
│   ├── app.py                    # Flask server (MAIN FILE)
│   ├── main.py                   # FastAPI alternative
│   ├── predict_terminal.py       # CLI prediction tool
│   └── requirements.txt          # Python dependencies
│
├── src/                          # ML preprocessing pipeline
│   ├── preprocess.py             # Data preprocessing
│   ├── LogisticRegression.py     # Model training
│   ├── Optimized.py              # Optimized model
│   └── Predict_Depression.py     # Prediction utilities
│
├── app.py                        # Root Flask app (for convenience)
├── best_model.pkl               # Trained ML model
├── Depression_Severity_Levels_Dataset.csv  # Training dataset
└── README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual Environment (venv)

### Installation

1. **Navigate to project directory:**
   ```bash
   cd c:\SocialMedia_Text_Depression
   ```

2. **Activate virtual environment:**
   ```bash
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install flask flask-cors nltk scikit-learn pandas numpy
   ```

### Running the Application

**Option 1: Run from backend folder (Recommended)**

```bash
cd backend
python app.py
```

**Option 2: Run from root directory**

```bash
python app.py
```

Both will start the server at: **http://localhost:5000**

---

## 💻 What You'll See

### Frontend Interface

1. **Header Section**
   - Title: "Depression Level Analyzer"
   - Calming design with soft colors

2. **Input Section**
   - Large textarea for user input
   - Character counter (max 1000 chars)
   - Prominent "Analyze My Text" button

3. **Results Section** (After analysis)
   - **Depression Severity**: Visual severity badge
   - **Risk Score**: Animated gauge (0-100)
   - **Emotion Detected**: Emoji + emotion type
   - **Crisis Alert**: For high-risk indicators
   - **Coping Strategies**: Personalized recommendations

---

## 🎨 Design Features

- **Color Palette**: Soft blues, lavenders, mints, greens
- **Typography**: Clean, readable fonts
- **Animations**: Smooth fade-in, bounce, slide effects
- **Responsive**: Works on desktop, tablet, mobile
- **Dark Mode**: Automatic system theme support
- **Accessibility**: Respects reduced-motion preferences

---

## 🔧 API Endpoints

### Health Check
```
GET /api/health
```
Returns: `{ status, model_loaded, vectorizer_loaded }`

### Prediction
```
POST /predict
Content-Type: application/x-www-form-urlencoded

text=<user_text>
```

**Response:**
```json
{
  "prediction": "Depression Detected 😔 (Confidence: 0.85)",
  "severity": "Moderate",
  "risk_score": 85.0,
  "suicide_risk": "High",
  "emotion": "Sadness",
  "strategies": [
    "Seek professional mental health support urgently",
    "Reach out to trusted friends or family members",
    ...
  ],
  "success": true
}
```

---

## 📊 Model Information

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF Vectorization (5000 features)
- **Training Data**: Depression Severity Levels Dataset
- **Preprocessing**: 
  - Emoji removal
  - Slang expansion (lol → laugh, u → you, etc.)
  - URL/email removal
  - Tokenization
  - Stopword removal
  - Lemmatization

---

## 🎯 Severity Levels

- **None**: No depression indicators
- **Mild**: Some depressive symptoms
- **Moderate**: Significant symptoms, professional help recommended
- **Severe**: Critical symptoms, urgent professional intervention needed

---

## 🆘 Crisis Resources

If you or someone you know is in crisis:

- **988 Suicide & Crisis Lifeline** (US): Call 988
- **Crisis Text Line** (US): Text HOME to 741741
- **International**: https://findahelpline.com

---

## 📝 Preprocessing Pipeline

1. Convert text to lowercase
2. Remove URLs
3. Remove email addresses
4. Remove emojis
5. Expand slangs (u→you, lol→laugh, etc.)
6. Remove special characters
7. Tokenization
8. Remove stopwords
9. Lemmatization

---

## 🛠️ Troubleshooting

### Port 5000 already in use?
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Model file not found?
Ensure `best_model.pkl` and `src/preprocessed_data.pkl` exist in the project root.

### CORS errors?
Flask-CORS is installed. Make sure you're running the latest `backend/app.py`.

---

## 📧 Support

For issues or questions, please refer to the source code comments or contact the development team.

---

## ⚖️ Disclaimer

This tool is for educational and awareness purposes only. It is **NOT** a substitute for professional mental health care. If you're experiencing depression, please consult with a licensed mental health professional.

---

**Made with ❤️ for mental health awareness**
