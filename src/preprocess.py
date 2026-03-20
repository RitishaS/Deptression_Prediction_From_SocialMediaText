import pandas as pd
import numpy as np
import re
import pickle
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load dataset
csv_path = "Depression_Severity_Levels_Dataset.csv"
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")

# Remove duplicates
print("\nRemoving duplicates...")
df = df.drop_duplicates(subset=['text'])
print(f"Dataset shape after removing duplicates: {df.shape}")

# Remove missing values
print("\nRemoving missing values...")
df = df.dropna(subset=['text', 'label'])
print(f"Dataset shape after removing NaNs: {df.shape}")

# Text preprocessing function
def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing
print("\nPreprocessing text...")
df['text'] = df['text'].apply(preprocess_text)

# Remove empty texts
df = df[df['text'].str.len() > 0]
print(f"Dataset shape after preprocessing: {df.shape}")

# Display sample
print("\nSample preprocessed data:")
print(df.head())

# Label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())

# TF-IDF Vectorization
print("\nApplying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    lowercase=True
)

X = vectorizer.fit_transform(df['text'].values)
y = df['label'].values

print(f"TF-IDF feature matrix shape: {X.shape}")
print(f"Number of features: {len(vectorizer.get_feature_names_out())}")

# Save preprocessed data
print("\nSaving preprocessed data and vectorizer...")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_file = os.path.join(script_dir, "preprocessed_data.csv")
pkl_file = os.path.join(script_dir, "preprocessed_data.pkl")

df.to_csv(csv_file, index=False)

# Save vectorizer and features
preprocessed_data = {
    'X': X.toarray(),
    'y': y,
    'vectorizer': vectorizer,
    'feature_names': vectorizer.get_feature_names_out()
}

with open(pkl_file, "wb") as f:
    pickle.dump(preprocessed_data, f)

print(f"Preprocessed data saved to {csv_file}")
print(f"Vectorizer saved to {pkl_file}")

print("\nPreprocessing complete!")
