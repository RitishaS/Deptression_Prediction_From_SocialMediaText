import pickle
import pandas as pd

print("\n" + "="*70)
print("DEPRESSION SEVERITY PREDICTION")
print("="*70)

# Load trained model
with open("improved_models.pkl", "rb") as f:
    model_package = pickle.load(f)

best_model = model_package['best_model']
best_model_name = model_package['best_model_name']
vectorizer = model_package['vectorizer']

print(f"\nLoaded model: {best_model_name}")
print(f"Model accuracy: {model_package['best_accuracy']:.4f} ({model_package['best_accuracy']*100:.2f}%)")

# Test predictions
print("\n" + "-"*70)
print("Test Predictions:")
print("-"*70)

test_texts = [
    "I feel so sad and hopeless, don't know what to do anymore",
    "Having a great day, life is wonderful!",
    "I'm a bit down but doing okay",
    "I want to end my life, suicide seems like the only option"
]

for text in test_texts:
    # Vectorize
    X = vectorizer.transform([text])
    
    # Predict
    prediction = best_model.predict(X)[0]
    
    # Get prediction probabilities (if available)
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(X)[0]
        confidence = proba.max()
    elif hasattr(best_model, 'decision_function'):
        decision = best_model.decision_function(X)[0]
        confidence = (decision - decision.min()) / (decision.max() - decision.min() + 1e-10)
        confidence = confidence.max()
    else:
        confidence = 0.0
    
    print(f"\nInput: \"{text[:60]}...\"")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")

print("\n" + "="*70)

# Interactive prediction
print("\nEnter your text (or 'quit' to exit):")
while True:
    user_input = input("\n> ")
    
    if user_input.lower() == 'quit':
        print("Exiting...")
        break
    
    # Vectorize
    X = vectorizer.transform([user_input])
    
    # Predict
    prediction = best_model.predict(X)[0]
    
    print(f"\nPredicted severity level: {prediction.upper()}")
    
    # Detailed output
    if prediction == "severe":
        print("⚠️  ALERT: Severe depression detected.")
        print("    Please seek immediate professional help.")
    elif prediction == "moderate":
        print("⚠️  WARNING: Moderate depression indicators.")
        print("    Consider consulting a mental health professional.")
    elif prediction == "mild":
        print("ℹ️  NOTICE: Mild depression symptoms detected.")
        print("    Regular self-care and support recommended.")
    else:  # minimum
        print("✓ Minimal depression indicators.")
        print("  Continue wellness practices.")

print("\n" + "="*70)
