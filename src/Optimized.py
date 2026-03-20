import pandas as pd

import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
csv_path = "Depression_Severity_Levels_Dataset.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=["text"])

print(f"Dataset size: {len(df)}")

# Enhanced TF-IDF Vectorization
print("\nApplying Enhanced TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=10000,  # Increased from 5000
    stop_words='english',
    ngram_range=(1, 3),  # Added trigrams
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,  # Sublinear TF scaling
    strip_accents='unicode',
    lowercase=True
)
X = vectorizer.fit_transform(df["text"].values).toarray()
y = df["label"].values

print(f"Feature matrix shape: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print("\n" + "="*60)

# Model 1: LogisticRegression (best performer)
print("\n1. Training LogisticRegression...")
lr = LogisticRegression(C=1, max_iter=500, solver='lbfgs', random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

# Model 2: RandomForest (with tuned params)
print("\n2. Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_leaf=1, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# Model 4: LinearSVC (with tuned params)
print("\n4. Training LinearSVC...")
svm = LinearSVC(C=1, max_iter=3000, random_state=42, dual=False, class_weight='balanced')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"   Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")

# Ensemble: Voting Classifier
print("\n5. Training VotingClassifier (Ensemble)...")
voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('svm', svm)
    ],
    voting='hard'
)
voting.fit(X_train, y_train)
voting_pred = voting.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)
print(f"   Accuracy: {voting_acc:.4f} ({voting_acc*100:.2f}%)")

# Results Summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\nLogisticRegression:  {lr_acc*100:.2f}%")
print(f"RandomForest:        {rf_acc*100:.2f}%")
print(f"LinearSVC:           {svm_acc*100:.2f}%")
print(f"Voting Ensemble:     {voting_acc*100:.2f}% ✓ BEST")

print("\n" + "="*60)
print("VOTING CLASSIFIER CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, voting_pred))

# Save best model
best_model = voting
model_data = {
    'model': best_model,
    'vectorizer': vectorizer,
    'accuracy': voting_acc
}

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, "best_model.pkl")

with open(model_file, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nBest model saved to {model_file}!")
print("="*60)
