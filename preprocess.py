import pandas as pd
import spacy
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────────────────────────
GLOVE_PATH    = "glove.6B.100d.txt"   # path to downloaded GloVe file
EMBEDDING_DIM = 100

# load dataset
df = pd.read_csv("Depression_Severity_Levels_Dataset.csv")

# load spacy model (keep tagger for POS annotations needed for lemmatization)
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "ner", "senter"])

# remove missing rows
df = df.dropna(subset=["text"])


def clean_text(text):
    # Regex cleaning
    text = re.sub(r'<[^>]+>', ' ', text)          # remove HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)   # remove URLs
    text = re.sub(r'@\w+', ' ', text)             # remove mentions
    text = re.sub(r'\d+', ' ', text)              # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)          # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()      # normalize spaces
    text = text.lower()
    return text


# Apply regex cleaning first
df["text"] = df["text"].apply(clean_text)

# Apply spaCy lemmatization in batches
clean_texts = []
for doc in nlp.pipe(df["text"], batch_size=128):
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 1]
    clean_texts.append(" ".join(tokens))

df["clean_text"] = clean_texts

print(df[["text", "clean_text"]].head())

# ── GloVe loading ─────────────────────────────────────────────────────────────
def load_glove(path):
    if not os.path.exists(path):
        print(f"Warning: GloVe file not found at '{path}'")
        print("To use GloVe embeddings, download glove.6B.100d.txt from:")
        print("https://nlp.stanford.edu/projects/glove/")
        print("and place it in the current directory.")
        return {}
    
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings):,} GloVe vectors (dim={EMBEDDING_DIM})")
    return embeddings


# ── Text → vector (mean pooling) ──────────────────────────────────────────────
def text_to_glove_vector(text, embeddings, dim):
    vectors = [embeddings[w] for w in text.split() if w in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(dim, dtype=np.float32)   # fallback for fully OOV documents


# ── Build feature matrix ──────────────────────────────────────────────────────
glove_embeddings = load_glove(GLOVE_PATH)

X = np.vstack([
    text_to_glove_vector(text, glove_embeddings, EMBEDDING_DIM)
    for text in df["clean_text"]
])
y = df["label"].values

print(f"Feature matrix shape : {X.shape}")
print(f"Label distribution   :\n{pd.Series(y).value_counts()}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

