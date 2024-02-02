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
        # Decode URL encoding
        url = unquote(url)
        
        # Remove http, https, www
        url = url.replace('http://', '').replace('https://', '').replace('www.', '')

        # Normalize Unicode characters to their ASCII equivalent
        url = unicodedata.normalize('NFKD', url).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # Convert all letters to lowercase
        url = url.lower()

        # Remove leading and trailing white spaces
        url = url.strip()

        # Replace IP addresses with a placeholder like "ipaddress"
        url = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'ipaddress', url)

        # Split URL based on delimiters
        tokens = re.split(r'[!$&\'()*+,;=:@._~:/?#\\[\]-]+', url)
