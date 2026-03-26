"""
Depression Text Classification Pipeline using GloVe Embeddings

This script trains and compares 3 ML models (SVM, Logistic Regression, Random Forest)
for depression detection using pre-trained GloVe word embeddings.

Pipeline:
1. Load and preprocess text data
2. Load GloVe embeddings
3. Convert text to GloVe vectors (sentence-level averaging)
4. Train 3 models
5. Evaluate and compare performance
6. Provide prediction function for new text
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import string
import os
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. TEXT PREPROCESSING
# ================================

def clean_text(text):
    """
    Clean and preprocess text:
    - Convert to lowercase
    - Remove URLs
    - Remove punctuation
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def tokenize_text(text):
    """
    Tokenize text into words.
    Remove punctuation but keep words.
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    words = text.split()
    
    return words

def preprocess_data(df):
    """
    Preprocess the entire dataset:
    - Clean text
    - Remove null values
    - Tokenize
    """
    print("📊 Preprocessing data...")
    
    # Remove null values
    df = df.dropna(subset=['text', 'label'])
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Tokenize
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    
    print(f"✓ Dataset size: {len(df)} samples")
    print(f"✓ Label distribution:\n{df['label'].value_counts()}\n")
    
    return df

# ================================
# 2. GLOVE EMBEDDINGS LOADING
# ================================

def load_glove_embeddings(glove_file_path=None, embedding_dim=100):
    """
    Load pre-trained GloVe embeddings.
    
    If glove_file_path is None, download from online source (requires internet).
    Otherwise, use the provided file path.
    
    Args:
        glove_file_path: Path to GloVe file (e.g., glove.6B.100d.txt)
        embedding_dim: Dimension of embeddings (100 is common)
    
    Returns:
        Dictionary: {word: vector}
    """
    print("📥 Loading GloVe embeddings...")
    
    embeddings = {}
    
    # Try to load from local file first
    if glove_file_path and os.path.exists(glove_file_path):
        print(f"📂 Loading from local file: {glove_file_path}")
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array([float(x) for x in values[1:]], dtype=np.float32)
                embeddings[word] = vector
    else:
        # Fallback: Create a simple GloVe-like embedding dictionary
        print("⚠️  GloVe file not found. Using random embeddings (for demo purposes).")
        print("📌 For production, download glove.6B.100d.txt from: https://nlp.stanford.edu/projects/glove/")
        print("   Extract and provide the file path.\n")
        
        # This is a placeholder for demonstration
        # In production, use actual pre-trained GloVe embeddings
        embeddings = create_demo_embeddings(embedding_dim)
    
    print(f"✓ Loaded {len(embeddings)} word embeddings")
    print(f"✓ Embedding dimension: {embedding_dim}\n")
    
    return embeddings, embedding_dim

def create_demo_embeddings(embedding_dim=100, vocab_size=10000):
    """
    Create random embeddings for demonstration.
    In production, use actual GloVe embeddings.
    """
    embeddings = {}
    np.random.seed(42)
    
    # Common words in depression texts
    common_words = [
        'depression', 'sad', 'anxious', 'happy', 'feel', 'feeling', 'think', 'thought',
        'help', 'support', 'struggle', 'difficult', 'pain', 'suffering', 'hope',
        'time', 'day', 'night', 'friend', 'family', 'love', 'lonely', 'alone',
        'cry', 'tears', 'stress', 'worry', 'fear', 'scared', 'bad', 'good',
        'better', 'worse', 'tired', 'sleep', 'wake', 'work', 'life', 'death',
        'suicide', 'crisis', 'help', 'therapy', 'doctor', 'medicine', 'medication',
        'professional', 'counselor', 'talk', 'listen', 'understand', 'care'
    ]
    
    for word in common_words:
        embeddings[word] = np.random.randn(embedding_dim).astype(np.float32)
    
    # Add random words for OOV handling
    for i in range(vocab_size - len(common_words)):
        word = f'word_{i}'
        embeddings[word] = np.random.randn(embedding_dim).astype(np.float32)
    
    return embeddings

# ================================
# 3. SENTENCE VECTORIZATION USING GLOVE
# ================================

def vectorize_sentence(tokens, embeddings, embedding_dim):
    """
    Convert a sentence (list of tokens) into a single fixed-size vector
    by averaging word embeddings.
    
    Args:
        tokens: List of words
        embeddings: Dictionary of word vectors
        embedding_dim: Dimension of embeddings
    
    Returns:
        np.array: Averaged word vector (fixed-size)
    """
    vectors = []
    
    for token in tokens:
        if token in embeddings:
            vectors.append(embeddings[token])
        # Unknown words are ignored (skip them)
        # Alternative: use zero vector with: else: vectors.append(np.zeros(embedding_dim))
    
    if len(vectors) == 0:
        # If no known words found, return zero vector
        return np.zeros(embedding_dim, dtype=np.float32)
    
    # Average all word vectors
    sentence_vector = np.mean(vectors, axis=0)
    
    return sentence_vector.astype(np.float32)

def create_feature_vectors(df, embeddings, embedding_dim):
    """
    Convert all texts to GloVe vectors.
    
    Returns:
        X: Feature matrix (n_samples, embedding_dim)
        y: Labels
    """
    print("🔄 Converting texts to GloVe vectors...")
    
    X = []
    for tokens in df['tokens']:
        vector = vectorize_sentence(tokens, embeddings, embedding_dim)
        X.append(vector)
    
    X = np.array(X)
    y = df['label'].values
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Feature dimension: {X.shape[1]}\n")
    
    return X, y

# ================================
# 4. MODEL TRAINING
# ================================

def train_svm(X_train, y_train):
    """Train Support Vector Machine"""
    print("🤖 Training SVM...")
    model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    model.fit(X_train, y_train)
    print("✓ SVM trained\n")
    return model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression"""
    print("🤖 Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    model.fit(X_train, y_train)
    print("✓ Logistic Regression trained\n")
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest"""
    print("🤖 Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✓ Random Forest trained\n")
    return model

# ================================
# 5. MODEL EVALUATION
# ================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model on test set and print metrics.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"📊 {model_name} - EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compare_models(results):
    """Compare all models and display rankings"""
    print(f"\n{'='*60}")
    print("🏆 MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    # Sort by F1-score
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print("Ranking by F1-Score:")
    print("-" * 60)
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank}. {result['model_name']}")
        print(f"   Accuracy:  {result['accuracy']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall:    {result['recall']:.4f}")
        print(f"   F1-Score:  {result['f1']:.4f}\n")
    
    best_model = sorted_results[0]
    print(f"🥇 Best Model: {best_model['model_name']} (F1: {best_model['f1']:.4f})")

# ================================
# 6. PREDICTION ON NEW TEXT
# ================================

def predict_depression(text, model, embeddings, embedding_dim, label_encoder):
    """
    Predict depression level for a new text input.
    
    Args:
        text: Input text to classify
        model: Trained model
        embeddings: GloVe embeddings dictionary
        embedding_dim: Embedding dimension
        label_encoder: Fitted label encoder
    
    Returns:
        Prediction and confidence score
    """
    # Preprocess
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    
    # Vectorize
    vector = vectorize_sentence(tokens, embeddings, embedding_dim)
    vector = vector.reshape(1, -1)
    
    # Predict
    prediction = model.predict(vector)[0]
    
    # Get confidence scores
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vector)[0]
        max_prob = max(probabilities)
    else:
        max_prob = None
    
    return prediction, max_prob

def interactive_prediction(model, embeddings, embedding_dim, label_encoder):
    """
    Interactive mode to test predictions on new text.
    """
    print(f"\n{'='*60}")
    print("🔮 INTERACTIVE PREDICTION MODE")
    print(f"{'='*60}")
    print("Enter text to classify (type 'exit' to quit):\n")
    
    while True:
        user_input = input("Your text: ").strip()
        
        if user_input.lower() == 'exit':
            print("✓ Exiting prediction mode.")
            break
        
        if not user_input:
            print("Please enter some text.\n")
            continue
        
        prediction, confidence = predict_depression(user_input, model, embeddings, embedding_dim, label_encoder)
        
        print(f"🎯 Prediction: {prediction}")
        if confidence:
            print(f"📊 Confidence: {confidence:.2%}")
        print()

# ================================
# 7. MAIN PIPELINE
# ================================

def main():
    """
    Main pipeline: Load data -> Preprocess -> Vectorize -> Train -> Evaluate -> Compare
    """
    print("\n" + "="*60)
    print("🧠 DEPRESSION TEXT CLASSIFICATION WITH GLOVE EMBEDDINGS")
    print("="*60 + "\n")
    
    # ========== STEP 1: LOAD DATA ==========
    print("Step 1️⃣ : Loading Data...")
    data_path = 'Depression_Severity_Levels_Dataset.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples\n")
    
    # ========== STEP 2: PREPROCESS ==========
    print("Step 2️⃣ : Preprocessing...")
    df = preprocess_data(df)
    
    # ========== STEP 3: LOAD GLOVE EMBEDDINGS ==========
    print("Step 3️⃣ : Loading GloVe Embeddings...")
    
    # Try to load from local file, else use demo
    glove_path = None  # Set to actual path if you have GloVe file
    embeddings, embedding_dim = load_glove_embeddings(glove_path, embedding_dim=100)
    
    # ========== STEP 4: VECTORIZE ==========
    print("Step 4️⃣ : Creating Feature Vectors...")
    X, y = create_feature_vectors(df, embeddings, embedding_dim)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Classes: {label_encoder.classes_}")
    print(f"✓ Encoded labels: {np.unique(y_encoded)}\n")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples\n")
    
    # ========== STEP 5: TRAIN MODELS ==========
    print("Step 5️⃣ : Training Models...")
    print("-" * 60)
    
    svm_model = train_svm(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # ========== STEP 6: EVALUATE ==========
    print("Step 6️⃣ : Evaluating Models...")
    
    results = []
    results.append(evaluate_model(svm_model, X_test, y_test, "Support Vector Machine (SVM)"))
    results.append(evaluate_model(lr_model, X_test, y_test, "Logistic Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))
    
    # ========== STEP 7: COMPARE ==========
    print("Step 7️⃣ : Comparing Models...")
    compare_models(results)
    
    # ========== STEP 8: INTERACTIVE PREDICTION ==========
    print("\n" + "="*60)
    decision = input("Do you want to test predictions on new text? (yes/no): ").strip().lower()
    
    if decision in ['yes', 'y']:
        best_model_name = max(results, key=lambda x: x['f1'])['model_name']
        
        if 'SVM' in best_model_name:
            best_model = svm_model
        elif 'Logistic' in best_model_name:
            best_model = lr_model
        else:
            best_model = rf_model
        
        interactive_prediction(best_model, embeddings, embedding_dim, label_encoder)
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
