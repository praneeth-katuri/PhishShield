import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, list):
            X = [X]

        features_list = [self.process_url(url) for url in X]
        return pd.DataFrame(features_list)

    """1. UsingIP : {-1,1}"""

    def using_ip(self, url):
        try:
            domain = urlparse(url).netloc
            ipaddress.ip_address(domain)
            return -1
        except ValueError:
            return 1

    """2. LongURL: {-1, 0, 1}"""

    def long_url(self, url):
        url_length = len(url)
        if url_length > 100:
            return -1
        elif url_length >= 50:
            return 0
        else:
            return 1
