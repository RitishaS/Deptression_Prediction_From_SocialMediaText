import pandas as pd
import spacy
import re

# load dataset
df = pd.read_csv("Depression_Severity_Levels_Dataset.csv")

# load spacy model
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

# remove null rows
df = df.dropna(subset=["text"])


# ---------- Regex Cleaning ----------
def clean_text(text):

    text = str(text)

    # remove HTML
    text = re.sub(r'<.*?>', ' ', text)

    # remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove mentions
    text = re.sub(r'@\w+', ' ', text)

    # normalize repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # lowercase
    text = text.lower()

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df["text"] = df["text"].apply(clean_text)


# ---------- spaCy processing ----------
clean_texts = []

for doc in nlp.pipe(df["text"], batch_size=256):

    tokens = []

    for token in doc:

        # ❗ DO NOT REMOVE stopwords
        # ❗ keep "not", "no", "i", etc.

        if token.is_punct:
            continue

        if len(token.text) == 1:
            continue

        tokens.append(token.lemma_)

    clean_texts.append(" ".join(tokens))


df["clean_text"] = clean_texts

print(df[["text","clean_text"]].head())

# save processed dataset
df.to_csv("processed_dataset.csv", index=False)

print("Preprocessing completed")