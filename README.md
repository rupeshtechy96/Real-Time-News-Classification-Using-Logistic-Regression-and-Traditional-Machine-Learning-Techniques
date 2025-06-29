# Real-Time-News-Classification-Using-Logistic-Regression-and-Traditional-Machine-Learning-Techniques
📌 Overview
This project focuses on building a machine learning pipeline to classify news articles in real-time into their respective categories using Logistic Regression and other traditional machine learning models. The goal is to automate the categorization process to help filter and organize digital news content effectively.
🎯 Objectives
*Automate classification of news articles into categories.
*Use traditional ML algorithms for accurate classification.
*Compare models based on multiple evaluation metrics.
*Simulate real-time news input and response.
🔍 Problem Statement
In today’s digital age, the sheer volume of online news makes manual classification impractical. This project leverages Natural Language Processing (NLP) and supervised machine learning to categorize articles automatically based on their textual content.
🧠 Techniques Used
Text Preprocessing
Tokenization
Stopword Removal
Lemmatization
Feature Extraction
Bag of Words (BoW)
TF-IDF Vectorization
Machine Learning Algorithms
Logistic Regression
Naive Bayes
Support Vector Machine (SVM)
Random Forest
Evaluation
Accuracy
Precision, Recall, F1-score
Confusion Matrix
📊 Dataset
Source: Kaggle Indexed News Dataset
Features: Fake news and Real news
Size: 25617 Articles
📈 Model Evaluation Results
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.9531	0.9464	0.9581	0.9522
Naive Bayes	0.8719	0.9819	0.7510	0.8511
SVM	0.9651	0.9649	0.9635	0.9642
Random Forest	0.9445	0.9577	0.9270	0.9421
Support Vector Machine (SVM) achieved the highest overall performance, followed closely by Logistic Regression.
🛠 Tools & Libraries
Language: Python 3.x
Libraries:
scikit-learn
pandas, numpy
nltk, spaCy
matplotlib, seaborn
joblib (for model serialization)
© 2025 Supriya Dutta and Rupesh Patel. All rights reserved. This project is developed and maintained by Supriya Dutta and Rupesh Patel. Unauthorized copying, reproduction, or redistribution is prohibited. This project is licensed under the MIT License - see the LICENSE file for details.
