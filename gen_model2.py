import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from feature_extraction.preprocess_url import URLPreprocessor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('datafiles/dataset_for_text_model.csv')

df = df.sample(frac=1).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(df.url, df.status)

pipeline = Pipeline([
    ('preprocessor', URLPreprocessor()),
    ('vectorizer', CountVectorizer(tokenizer=None, stop_words=None, lowercase=False, ngram_range=(1, 2))),
    ('classifier', LinearRegression())
])

print("started")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_train)

pipeline.score(X_test, y_test)
import pickle

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

