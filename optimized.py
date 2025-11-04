# -----------------------------------------------
# CASE STUDY: Sentiment Analysis (Restaurant Reviews)
# Goal: Handle underfitting, compare models, check bias/variance/accuracy/AUC
# -----------------------------------------------

import numpy as np
import pandas as pd
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv(r'C:\Users\aashutosh\Downloads\Restaurant_Reviews.tsv' ,delimiter = '\t', quoting = 3)

# -----------------------------------------------
# Optional: Increase data by duplicating rows
dataset = pd.concat([dataset] * 3, ignore_index=True)  # 3x larger dataset

# -----------------------------------------------
# Preprocessing function
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_text(remove_stopwords=True):
    corpus = []
    lemmatizer = WordNetLemmatizer()
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        if remove_stopwords:
            review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
        else:
            review = [lemmatizer.lemmatize(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

# -----------------------------------------------
# Vectorizer Functions (BoW & TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_vectors(corpus, vectorizer_type='tfidf'):
    if vectorizer_type == 'bow':
        vectorizer = CountVectorizer(max_features=2500, ngram_range=(1,2))
    else:
        vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus).toarray()
    return X

# -----------------------------------------------
# Train-test split
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.25):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# -----------------------------------------------
# Model evaluation metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except:
        # For models without predict_proba
        y_prob = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = None
    bias = model.score(X_train, y_train)
    variance = model.score(X_test, y_test)

    return accuracy, auc, bias, variance

# -----------------------------------------------
# All classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# -----------------------------------------------
# Experiment configurations
configs = [
    ("BoW", True),
    ("TFIDF", True),
    ("BoW", False),
    ("TFIDF", False)
]

y = dataset.iloc[:, 1].values

# -----------------------------------------------
# Run experiments
results = []

for vec_type, stopwords_option in configs:
    print(f"\n==============================")
    print(f"Vectorizer: {vec_type}, Stopwords Removed: {stopwords_option}")
    print(f"==============================")
    
    corpus = clean_text(remove_stopwords=stopwords_option)
    X = get_vectors(corpus, vectorizer_type=vec_type.lower())
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)

    for name, model in models.items():
        try:
            acc, auc, bias, var = evaluate_model(model, X_train, X_test, y_train, y_test)
            results.append([vec_type, stopwords_option, name, acc, auc, bias, var])
            print(f"{name:20s} | Acc: {acc*100:.2f}% | AUC: {auc:.3f} | Bias: {bias*100:.2f}% | Var: {var*100:.2f}%")
        except Exception as e:
            print(f"{name:20s} | Error: {e}")

# -----------------------------------------------
# Final Summary
results_df = pd.DataFrame(results, columns=["Vectorizer", "Stopwords_Removed", "Model", "Accuracy", "AUC", "Bias", "Variance"])

print("\n\n==============================")
print("FINAL COMPARISON TABLE")
print("==============================")
print(results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True))

