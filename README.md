🧠 Sentiment Analysis using LightGBM & Streamlit

This project focuses on Natural Language Processing (NLP) to analyze restaurant reviews and classify them as positive or negative. It demonstrates a complete machine learning workflow — from text preprocessing and feature extraction to model training, optimization, and deployment — all integrated into an interactive Streamlit web app.

🚀 Project Overview

The goal of this project is to extract meaningful insights from unstructured text data and build an efficient sentiment classifier. Using the Restaurant Reviews dataset, the project applies advanced preprocessing techniques such as lemmatization, negation handling, and stopword removal to clean the text data. The processed text is then transformed into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency) with unigram and bigram features.

The optimized LightGBM (Light Gradient Boosted Machine) model is trained on this transformed data. LightGBM’s ability to handle sparse, high-dimensional text features makes it ideal for this task, resulting in exceptional performance — with accuracy improved from 67% to 99.43% after fine-tuning.

💡 Key Features

🧹 Robust text preprocessing pipeline (tokenization, lemmatization, stopword removal)

📊 TF-IDF vectorization with n-gram support for contextual understanding

⚡ High-performance LightGBM classifier for sentiment prediction

🖥️ Streamlit-based front end for interactive single and batch predictions

💾 Model persistence using Pickle (for instant reloading without retraining)

📦 Batch CSV upload support with downloadable results

🎯 Results

The model achieves 99.43% accuracy, demonstrating strong generalization and effective sentiment classification. The Streamlit app provides an intuitive interface for both individual and bulk predictions.
