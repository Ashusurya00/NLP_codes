# -*- coding: utf-8 -*-
"""
High-Accuracy Sentiment Classifier with Enhanced Features & Hyperparameter Tuning
@author: Ashutosh
"""

import pandas as pd
import numpy as np
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# ====================== LOAD DATA ======================
dataset = pd.read_csv(r'C:\Users\aashutosh\Downloads\Restaurant_Reviews.tsv',
                      delimiter='\t', quoting=3)

# ====================== TEXT CLEANING ======================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

corpus = [clean_text(review) for review in dataset['Review']]

# ====================== ADVANCED FEATURE EXTRACTION ======================
from sklearn.feature_extraction.text import TfidfVectorizer

# Increase vocabulary & use trigrams for more context
tfidf = TfidfVectorizer(
    max_features=5000,      # increased from 1500 to 5000
    ngram_range=(1, 3),     # include up to 3-word phrases
    min_df=2,               # remove rare words
    max_df=0.8,             # remove too frequent words
    sublinear_tf=True       # smooth term frequencies
)

X = tfidf.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# ====================== SPLIT ======================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ====================== MODEL CANDIDATES ======================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (RBF)': SVC(kernel='rbf'),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ====================== BASELINE COMPARISON ======================
from sklearn.metrics import accuracy_score, f1_score

print("🔹 Baseline Model Comparison")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: Accuracy = {accuracy_score(y_test, y_pred):.4f}, F1 = {f1_score(y_test, y_pred):.4f}")

# ====================== HYPERPARAMETER TUNING (BEST MODEL) ======================
from sklearn.model_selection import GridSearchCV

# Usually, SVM or XGBoost will perform best on text data
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(),
                    param_grid,
                    scoring='accuracy',
                    cv=5,
                    verbose=1,
                    n_jobs=-1)

grid.fit(X_train, y_train)

print("\n✅ Best Parameters for SVM:", grid.best_params_)
best_model = grid.best_estimator_

# ====================== FINAL EVALUATION ======================
y_pred_final = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)

print(f"\n🎯 Final Model: SVM (Optimized)")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# ====================== CONFUSION MATRIX ======================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Best Model")
plt.show()

# ====================== TRAINING SCORE (BIAS CHECK) ======================
train_acc = accuracy_score(y_train, best_model.predict(X_train))
print(f"\nTraining Accuracy (for bias check): {train_acc:.4f}")
