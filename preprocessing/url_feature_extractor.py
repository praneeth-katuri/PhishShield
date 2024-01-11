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

    """3. ShortURL: {-1, 1}"""

    def short_url(self, url):
        shortening_domains = [
            "bit.ly",
            "goo.gl",
            "shorte.st",
            "go2l.ink",
            "x.co",
            "ow.ly",
            "t.co",
            "tinyurl",
            "tr.im",
            "is.gd",
            "cli.gs",
            "yfrog.com",
            "migre.me",
            "ff.im",
            "tiny.cc",
            "url4.eu",
            "twit.ac",
            "su.pr",
            "twurl.nl",
            "snipurl.com",
            "short.to",
            "BudURL.com",
            "ping.fm",
            "post.ly",
            "Just.as",
            "bkite.com",
            "snipr.com",
            "fic.kr",
            "loopt.us",
            "doiop.com",
            "short.ie",
            "kl.am",
            "wp.me",
            "rubyurl.com",
            "om.ly",
            "to.ly",
            "bit.do",
            "t.co",
            "lnkd.in",
            "db.tt",
            "qr.ae",
            "adf.ly",
            "goo.gl",
            "bitly.com",
            "cur.lv",
            "tinyurl.com",
            "ow.ly",
            "bit.ly",
            "ity.im",
            "q.gs",
            "is.gd",
            "po.st",
            "bc.vc",
            "twitthis.com",
            "u.to",
            "j.mp",
            "buzurl.com",
            "cutt.us",
            "u.bb",
            "yourls.org",
            "x.co",
            "prettylinkpro.com",
            "scrnch.me",
            "filoops.info",
            "vzturl.com",
            "qr.net",
            "1url.com",
            "tweez.me",
            "v.gd",
            "tr.im",
            "link.zip.net",
        ]
        return -1 if any(domain in url for domain in shortening_domains) else 1

    """4. Symbol@: {-1,1}"""

    def symbol_at(self, url):
        parsed_url = urlparse(url)
        return -1 if "@" in parsed_url else 1
