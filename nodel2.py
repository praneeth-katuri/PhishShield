import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import time
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame containing the data.
    """
    return pd.read_csv(file_path)

def tokenize_text(data, column):
    """
    Tokenize text data in a DataFrame column.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - column (str): Name of the column containing text data.
    """
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    data['text_tokenized'] = data[column].map(lambda t: tokenizer.tokenize(t))

def stem_text(data):
    """
    Stem tokenized text data in a DataFrame.

    Args:
    - data (pd.DataFrame): DataFrame containing the tokenized text data.
    """
    stemmer = SnowballStemmer("english")
    data['text_stemmed'] = data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])

def join_text(data):
    """
    Join stemmed text data in a DataFrame.

    Args:
    - data (pd.DataFrame): DataFrame containing the stemmed text data.
    """
    data['text_sent'] = data['text_stemmed'].map(lambda l: ' '.join(l))

def vectorize_text(data):
    """
    Vectorize text data using CountVectorizer.

    Args:
    - data (pd.DataFrame): DataFrame containing the text data.

    Returns:
    - scipy.sparse.csr_matrix: Sparse matrix of token counts.
    """
    count_vectorizer = CountVectorizer()
    features = count_vectorizer.fit_transform(data['text_sent'])
    return features

def evaluate_model(model, features, labels):
    """
    Evaluate a trained model.

    Args:
    - model (sklearn.base.BaseEstimator): Trained model.
    - features (scipy.sparse.csr_matrix): Sparse matrix of features.
    - labels (pd.Series): Series containing the labels.
    """
    print('Training Accuracy:', model.score(features, labels))
    predictions = model.predict(features)
    conf_matrix = confusion_matrix(predictions, labels)
    print('\nCLASSIFICATION REPORT\n')
    print(classification_report(predictions, labels, target_names=['Bad', 'Good']))
    print('\nCONFUSION MATRIX')
    print(conf_matrix)

def main():
    """
    Main function to execute the entire workflow.
    """
    warnings.filterwarnings('ignore')
    phish_data = load_data('datafiles/dataset_for_text_model.csv')

    tokenize_text(phish_data, 'url')
    stem_text(phish_data)
    join_text(phish_data)

    features = vectorize_text(phish_data)
    labels = phish_data['status']

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    pipeline_ls = make_pipeline(CountVectorizer(tokenizer=RegexpTokenizer(r'[A-Za-z]+').tokenize, stop_words='english'),
                                LogisticRegression())
    pipeline_ls.fit(train_features, train_labels)

    print('Training Accuracy:', pipeline_ls.score(train_features, train_labels))
    print('Testing Accuracy:', pipeline_ls.score(test_features, test_labels))
    evaluate_model(pipeline_ls, test_features, test_labels)

    pickle.dump(pipeline_ls, open('pickle/text_model.pkl', 'wb'))

if __name__ == "__main__":
    main()
