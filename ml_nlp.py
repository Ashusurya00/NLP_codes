# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 19:30:27 2025

@author: aashutosh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\aashutosh\Downloads\Restaurant_Reviews.tsv' ,delimiter = '\t', quoting = 3)

import re 
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])   # Remove non-letter characters
    review = review.lower()                           # Convert to lowercase
    review = review.split()
    ps = PorterStemmer()                           # Tokenize into words
    #review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]  # ✅ Corrected line
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

print(ac)

bias = classifier.score(X_train, y_train) 
bias

variance = classifier.score(X_test, y_test)
variance
