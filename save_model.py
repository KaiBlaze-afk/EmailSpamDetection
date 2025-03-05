import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("mail_data.csv").fillna('')

X = df['Message']
y = df['Category'].map(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

with open('./spam_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("Model saved successfully!")
