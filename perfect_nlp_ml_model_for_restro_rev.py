# -*- coding: utf-8 -*-
"""
Sentiment Analysis (Restaurant Reviews)
Optimized version using only LightGBM
@author: Ashutosh
"""

import numpy as np
import pandas as pd
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv(r'C:\Users\aashutosh\Downloads\Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Optional: expand data to balance classes (comment out if slow)
dataset = pd.concat([dataset] * 36, ignore_index=True)

# -----------------------------------------------
# Preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_text(texts, remove_stopwords=True):
    lemmatizer = WordNetLemmatizer()
    stop_words_set = set(stopwords.words('english'))
    corpus = []
    for text in texts:
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        if remove_stopwords:
            review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words_set]
        else:
            review = [lemmatizer.lemmatize(word) for word in review]
        # Simple negation handling
        for j in range(len(review) - 1):
            if review[j] == 'not':
                review[j + 1] = 'not_' + review[j + 1]
        corpus.append(' '.join(review))
    return corpus

print("Cleaning text...")
corpus = clean_text(dataset['Review'])

# -----------------------------------------------
# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# -----------------------------------------------
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# -----------------------------------------------
# LightGBM Model
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=31,
    random_state=0
)

print("\nTraining LightGBM model...")
model.fit(X_train, y_train)

# -----------------------------------------------
# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("🔹 LightGBM Model Performance 🔹")
print("==============================")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"✅ F1 Score: {f1 * 100:.2f}%")
print(f"✅ AUC Score: {auc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n✅ Model training & evaluation completed successfully.")
