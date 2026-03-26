"""
Advanced Usage Examples - GloVe Depression Classifier

This file shows how to use individual functions from the main classifier
for custom workflows and advanced customization.
"""

from GloVe_Depression_Classifier import (
    clean_text, tokenize_text, preprocess_data, load_glove_embeddings,
    create_feature_vectors, vectorize_sentence, train_svm, train_logistic_regression,
    train_random_forest, evaluate_model, predict_depression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ================================
# EXAMPLE 1: Custom Preprocessing
# ================================

def example_1_custom_preprocessing():
    """Example: Load data with custom preprocessing"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Custom Text Preprocessing")
    print("="*60 + "\n")
    
    # Sample text
    texts = [
        "I'm feeling really sad and depressed lately...",
        "Check out https://example.com for help!!!",
        "Email me at test@email.com for support"
    ]
    
    for text in texts:
        cleaned = clean_text(text)
        tokens = tokenize_text(cleaned)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Tokens:   {tokens}\n")

# ================================
# EXAMPLE 2: Manual Vectorization
# ================================

def example_2_manual_vectorization():
    """Example: Manually vectorize text using GloVe"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Manual Text Vectorization with GloVe")
    print("="*60 + "\n")
    
    # Load minimal embeddings
    embeddings, embedding_dim = load_glove_embeddings(embedding_dim=100)
    
    # Example text
    text = "I feel depressed and sad"
    print(f"Text: '{text}'")
    print(f"Embedding dimension: {embedding_dim}\n")
    
    # Process
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    print(f"Tokens: {tokens}")
    
    # Vectorize
    vector = vectorize_sentence(tokens, embeddings, embedding_dim)
    print(f"Vector shape: {vector.shape}")
    print(f"Vector values (first 10): {vector[:10]}")
    print(f"Vector mean: {np.mean(vector):.4f}")
    print(f"Vector std: {np.std(vector):.4f}\n")

# ================================
# EXAMPLE 3: Train Only One Model
# ================================

def example_3_single_model():
    """Example: Train and test only one specific model"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Training Single Model (SVM)")
    print("="*60 + "\n")
    
    # Load data
    df = pd.read_csv('Depression_Severity_Levels_Dataset.csv')
    df = preprocess_data(df)
    
    # Vectorize
    embeddings, embedding_dim = load_glove_embeddings(embedding_dim=100)
    X, y = create_feature_vectors(df, embeddings, embedding_dim)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train only SVM
    print("Training Support Vector Machine with custom parameters...")
    svm_model = train_svm(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(svm_model, X_test, y_test, "SVM (Custom)")

# ================================
# EXAMPLE 4: Compare Two Models Only
# ================================

def example_4_compare_two_models():
    """Example: Train and compare only 2 models"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Comparing Two Models")
    print("="*60 + "\n")
    
    # Load and prepare data
    df = pd.read_csv('Depression_Severity_Levels_Dataset.csv')
    df = preprocess_data(df)
    
    embeddings, embedding_dim = load_glove_embeddings(embedding_dim=100)
    X, y = create_feature_vectors(df, embeddings, embedding_dim)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train 2 models
    print("Training models...\n")
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate both
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Quick comparison
    print(f"\n🏆 Winner: {'LR' if lr_results['f1'] > rf_results['f1'] else 'RF'}")

# ================================
# EXAMPLE 5: Batch Prediction
# ================================

def example_5_batch_prediction():
    """Example: Predict on multiple texts at once"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Prediction on Multiple Texts")
    print("="*60 + "\n")
    
    # Load data and train model
    df = pd.read_csv('Depression_Severity_Levels_Dataset.csv')
    df = preprocess_data(df)
    
    embeddings, embedding_dim = load_glove_embeddings(embedding_dim=100)
    X, y = create_feature_vectors(df, embeddings, embedding_dim)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    model = train_logistic_regression(X_train, y_train)
    
    # Test texts
    test_texts = [
        "I'm feeling great today, everything is wonderful!",
        "I can't stop thinking negative thoughts, everything feels hopeless",
        "I'm okay, just a bit tired",
        "Life has lost all meaning, I don't see the point anymore"
    ]
    
    print("Predictions on test texts:\n")
    for text in test_texts:
        prediction, confidence = predict_depression(text, model, embeddings, embedding_dim, label_encoder)
        print(f"Text: '{text}'")
        print(f"  → Prediction: {prediction}")
        if confidence:
            print(f"  → Confidence: {confidence:.2%}\n")

# ================================
# EXAMPLE 6: Different Embedding Dimensions
# ================================

def example_6_embedding_dimensions():
    """Example: Compare different embedding dimensions"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Impact of Different Embedding Dimensions")
    print("="*60 + "\n")
    
    df = pd.read_csv('Depression_Severity_Levels_Dataset.csv')
    df = preprocess_data(df)
    
    # Note: In real GloVe files, you can use 50d or 300d
    # For demo, we'll just show the concept
    
    print("Available GloVe dimension options:")
    print("  - 50d  : Faster, lower quality")
    print("  - 100d : Balanced (current default)")
    print("  - 200d : Better quality, slower")
    print("  - 300d : Best quality, much slower")
    print("\nUsage:")
    print("  embeddings, dim = load_glove_embeddings(glove_file, embedding_dim=300)")

# ================================
# EXAMPLE 7: Cross-Validation
# ================================

def example_7_cross_validation():
    """Example: Manual cross-validation for robustness"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Manual 5-Fold Cross-Validation")
    print("="*60 + "\n")
    
    from sklearn.model_selection import StratifiedKFold
    
    # Load and prepare
    df = pd.read_csv('Depression_Severity_Levels_Dataset.csv')
    df = preprocess_data(df)
    
    embeddings, embedding_dim = load_glove_embeddings(embedding_dim=100)
    X, y = create_feature_vectors(df, embeddings, embedding_dim)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    print("Running 5-Fold Cross-Validation...\n")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]
        
        # Train
        model = train_logistic_regression(X_train_fold, y_train_fold)
        
        # Evaluate
        from sklearn.metrics import f1_score
        y_pred = model.predict(X_test_fold)
        f1 = f1_score(y_test_fold, y_pred, average='weighted', zero_division=0)
        fold_scores.append(f1)
        
        print(f"Fold {fold} F1-Score: {f1:.4f}")
    
    print(f"\nMean F1-Score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# ================================
# EXAMPLE 8: Hyperparameter Tuning
# ================================

def example_8_hyperparameter_tuning():
    """Example: Grid search for best SVM parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Hyperparameter Tuning for SVM")
    print("="*60 + "\n")
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    
    # Load and prepare
    df = pd.read_csv('Depression_Severity_Levels_Dataset.csv')
    df = preprocess_data(df)
    
    embeddings, embedding_dim = load_glove_embeddings(embedding_dim=100)
    X, y = create_feature_vectors(df, embeddings, embedding_dim)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Grid search
    print("Finding best SVM parameters (this may take a while)...\n")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    # Test on test set
    test_score = grid_search.score(X_test, y_test)
    print(f"Test set accuracy: {test_score:.4f}")

# ================================
# RUN EXAMPLES
# ================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 ADVANCED USAGE EXAMPLES")
    print("="*60)
    
    # Uncomment the example you want to run
    
    example_1_custom_preprocessing()
    example_2_manual_vectorization()
    # example_3_single_model()
    # example_4_compare_two_models()
    # example_5_batch_prediction()
    # example_6_embedding_dimensions()
    # example_7_cross_validation()
    # example_8_hyperparameter_tuning()
    
    print("\n✅ Examples completed!")
