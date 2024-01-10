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
