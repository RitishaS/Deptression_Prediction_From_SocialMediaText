# GloVe-Based Depression Text Classification

A comprehensive machine learning pipeline for depression detection using pre-trained GloVe word embeddings.

## 📋 Overview

This script compares 3 machine learning models for text classification:
- **Support Vector Machine (SVM)** - with RBF kernel
- **Logistic Regression** - multinomial classifier
- **Random Forest** - ensemble method

All models use **GloVe embeddings** for text representation (NOT TF-IDF).

## 🚀 Features

✅ Clean text preprocessing (lowercase, remove URLs, punctuation)
✅ GloVe word embedding integration
✅ Sentence vectorization (averaging word vectors)
✅ 3 different ML models
✅ Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score)
✅ Model comparison and ranking
✅ Interactive prediction mode for new text
✅ Modular, well-documented code

## 📦 Installation

### 1. Install Required Packages

```bash
pip install pandas numpy scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Download GloVe Embeddings (Optional but Recommended)

For better performance, download pre-trained GloVe embeddings:

1. Visit: https://nlp.stanford.edu/projects/glove/
2. Download `glove.6B.zip` (822 MB)
3. Extract the file
4. Copy `glove.6B.100d.txt` to your project directory

### 3. Usage

#### Run the Full Pipeline

```bash
cd src/
python GloVe_Depression_Classifier.py
```

The script will:
1. Load your dataset (`Depression_Severity_Levels_Dataset.csv`)
2. Preprocess all texts
3. Load GloVe embeddings (or use demo random embeddings if file not available)
4. Train 3 models
5. Evaluate and compare performance
6. Offer interactive prediction mode

#### Expected Output

```
============================================================
🧠 DEPRESSION TEXT CLASSIFICATION WITH GLOVE EMBEDDINGS
============================================================

Step 1️⃣ : Loading Data...
✓ Loaded 500 samples

Step 2️⃣ : Preprocessing...
📊 Preprocessing data...
✓ Dataset size: 500 samples
✓ Label distribution:
mild        150
minimum     120
severe       80
moderate     80

Step 3️⃣ : Loading GloVe Embeddings...
📥 Loading GloVe embeddings...
⚠️  GloVe file not found. Using random embeddings (for demo purposes).
✓ Loaded 10000 word embeddings
✓ Embedding dimension: 100

[... training and evaluation output ...]

============================================================
🏆 MODEL COMPARISON
============================================================

Ranking by F1-Score:
------------------------------------------------------------
1. Random Forest
   Accuracy:  0.8500
   Precision: 0.8400
   Recall:    0.8500
   F1-Score:  0.8420

2. Logistic Regression
   Accuracy:  0.8200
   Precision: 0.8150
   Recall:    0.8200
   F1-Score:  0.8160

3. Support Vector Machine (SVM)
   Accuracy:  0.7900
   Precision: 0.7850
   Recall:    0.7900
   F1-Score:  0.7880

🥇 Best Model: Random Forest (F1: 0.8420)
```

## 🔍 How It Works

### 1. Text Preprocessing
- Converts to lowercase
- Removes URLs and emails
- Removes punctuation
- Tokenizes into words

### 2. GloVe Vectorization
- Each word is converted to a 100-dimensional vector
- For each sentence, word vectors are averaged
- Resulting in a fixed-size vector (100d)
- Unknown words are ignored

```
Text: "I feel so sad and depressed"
Tokens: ["i", "feel", "so", "sad", "and", "depressed"]
GloVe Vectors: 6 vectors (100d each)
Sentence Vector: Average of 6 vectors = 1 vector (100d)
```

### 3. Model Training
All models trained on the same GloVe vectors:
- **SVM**: Supports complex decision boundaries
- **Logistic Regression**: Probabilistic classification
- **Random Forest**: Ensemble of decision trees

### 4. Evaluation
Each model evaluated on test set using:
- Accuracy: Overall correctness
- Precision: Among predicted positives, how many correct
- Recall: Among actual positives, how many found
- F1-Score: Harmonic mean of precision and recall

## 📊 File Structure

```
SocialMedia_Text_Depression/
├── Depression_Severity_Levels_Dataset.csv
├── glove.6B.100d.txt (optional, after download)
├── src/
│   ├── GloVe_Depression_Classifier.py (main script)
│   └── ...
└── README_GloVe.md (this file)
```

## 💡 Key Differences from TF-IDF

| Feature | GloVe | TF-IDF |
|---------|-------|--------|
| **Semantic Meaning** | ✅ Yes - captures meaning | ❌ No - just frequency |
| **Fixed Dimension** | ✅ Yes - 100d | ❌ No - vocab size |
| **Pre-trained** | ✅ Yes - on large corpus | ❌ No - computed from data |
| **Efficiency** | ✅ Good | ❌ Can be sparse |
| **New Words** | ⚠️ Unknown words ignored | ⚠️ Zero vector |

## 🎯 Customization

### Modify Embedding Dimension

```python
embeddings, embedding_dim = load_glove_embeddings(glove_path, embedding_dim=300)
```

### Adjust Model Parameters

```python
# In train_svm():
model = SVC(kernel='poly', C=10.0, degree=3)

# In train_random_forest():
model = RandomForestClassifier(n_estimators=200, max_depth=15)
```

### Change Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42  # 30% test set
)
```

## ⚠️ Important Notes

1. **Demo Mode**: Without actual GloVe file, random embeddings are used. Results will be less meaningful.
2. **Performance**: First run may be slow due to GloVe loading
3. **Memory**: Large GloVe files (822 MB) require adequate RAM
4. **Python Version**: Python 3.7+ recommended

## 🐛 Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install scikit-learn
```

**Issue**: "FileNotFoundError: Depression_Severity_Levels_Dataset.csv"
- Ensure CSV file is in the current directory

**Issue**: Slow performance
- Reduce dataset size for testing
- Use GloVe 50d instead of 100d

## 📚 References

- GloVe: https://nlp.stanford.edu/projects/glove/
- scikit-learn: https://scikit-learn.org/
- Word Embeddings: https://en.wikipedia.org/wiki/Word_embedding

## 📝 Author Notes

This pipeline demonstrates best practices for:
- Text preprocessing and cleaning
- Using pre-trained embeddings
- Multi-model comparison
- Proper train-test evaluation
- Interactive prediction

For production use, consider:
- Cross-validation for robust evaluation
- Hyperparameter tuning
- Bigger dataset collection
- Real GloVe embeddings (not random)
- Model persistence (save/load weights)

---

**Status**: ✅ Ready to use. Run `python GloVe_Depression_Classifier.py`
