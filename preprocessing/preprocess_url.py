import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import unquote
from sklearn.base import BaseEstimator, TransformerMixin

class URLPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_url(url) for url in X]

    def preprocess_url(self, url):
        
