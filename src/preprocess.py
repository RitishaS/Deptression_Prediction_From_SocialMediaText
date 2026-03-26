import pandas as pd
import numpy as np
import re
import pickle
import os
import sys
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Compile regex patterns once at module level (performance optimization)
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
EMAIL_PATTERN = re.compile(r'\S+@\S+')
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
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
    "\ufe0f"  # dingbats
    "\u3030"
    "]+",
    flags=re.UNICODE
)
SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')

# Initialize once at module level (performance optimization)
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# Slang dictionary for social media text
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

# Configuration
CONFIG = {
    'csv_path': 'Depression_Severity_Levels_Dataset.csv',
    'tfidf_max_features': 5000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.95,
    'min_token_length': 2,
}

def preprocess_text(text):
    """
    Clean and preprocess text data with optimizations.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
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
    
    # Replace slangs with their meanings
    words = text.split()
    words = [SLANG_DICT.get(word, word) for word in words]
    text = ' '.join(words)
    
    # Remove special characters and digits
    text = SPECIAL_CHAR_PATTERN.sub('', text)
    
    # Remove extra whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [
        word for word in tokens 
        if word not in STOP_WORDS and len(word) > CONFIG['min_token_length']
    ]
    
    # Lemmatization
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def load_dataset(csv_path):
    """Load and validate dataset."""
    try:
        logger.info(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        sys.exit(1)

def clean_dataset(df):
    """Remove duplicates and missing values."""
    logger.info("\nRemoving duplicates...")
    df = df.drop_duplicates(subset=['text'])
    logger.info(f"Dataset shape after removing duplicates: {df.shape}")
    
    logger.info("Removing missing values...")
    df = df.dropna(subset=['text', 'label'])
    logger.info(f"Dataset shape after removing NaNs: {df.shape}")
    
    return df

def apply_preprocessing(df):
    """Apply preprocessing to text column."""
    logger.info("\nPreprocessing text...")
    df['text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    logger.info(f"Dataset shape after preprocessing: {df.shape}")
    
    return df

def vectorize_data(df):
    """Apply TF-IDF vectorization."""
    logger.info("\nApplying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG['tfidf_max_features'],
        stop_words='english',
        ngram_range=CONFIG['tfidf_ngram_range'],
        min_df=CONFIG['tfidf_min_df'],
        max_df=CONFIG['tfidf_max_df'],
        sublinear_tf=True,
        lowercase=True
    )
    
    X = vectorizer.fit_transform(df['text'].values)
    y = df['label'].values
    
    logger.info(f"TF-IDF feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    
    return X, y, vectorizer

def display_statistics(df):
    """Display data statistics."""
    logger.info("\nSample preprocessed data:")
    logger.info(df.head())
    logger.info("\nLabel distribution:")
    logger.info(df['label'].value_counts())


def save_data(df, output_dir):
    """Save preprocessed data (text and label only) to CSV."""
    try:
        logger.info("\nSaving preprocessed data (text and label only)...")
        csv_file = os.path.join(output_dir, "preprocessed_data_no_tfidf.csv")
        pkl_file = os.path.join(output_dir, "preprocessed_data_no_tfidf.pkl")
        # Save CSV
        df[['text', 'label']].to_csv(csv_file, index=False)
        # Save as pickle (dict with text and label)
        preprocessed_data = {
            'text': df['text'].tolist(),
            'label': df['label'].tolist()
        }
        with open(pkl_file, "wb") as f:
            pickle.dump(preprocessed_data, f)
        logger.info(f"Preprocessed data saved to {csv_file}")
        logger.info(f"Preprocessed pickle saved to {pkl_file}")
    except IOError as e:
        logger.error(f"Error saving data: {e}")
        sys.exit(1)

def main():
    """Main preprocessing pipeline."""
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(os.path.dirname(script_dir), CONFIG['csv_path'])
        
        # Load and process data
        df = load_dataset(csv_path)
        df = clean_dataset(df)
        df = apply_preprocessing(df)
        
        # Display statistics
        display_statistics(df)
        

        # Save only preprocessed text and label (no TF-IDF)
        save_data(df, script_dir)
        logger.info("\nPreprocessing complete! (No TF-IDF)")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()