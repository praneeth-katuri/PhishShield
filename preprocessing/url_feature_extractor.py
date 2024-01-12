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

    """5. Redirecting//: {-1, 1}"""
    def double_slash_redirecting(self, url):
        parsed_url = urlparse(url)
        return -1 if "//" in parsed_url else 1

    '''6. PrefixSuffix-: {-1,1}'''
    def prefix_suffix(self, url):
        domain = urlparse(url).netloc.split("/")[0].split('?')[0].split('#')[0]
        return -1 if '-' in domain else 1
    
        '''7. SubDomains: {-1, 0, 1}'''
    def sub_domains(self, url):
        try:
            parsed_url = urlparse(url)
            netloc_parts = parsed_url.netloc.split('.')

            # Omitting the first "www." part if present
            if netloc_parts[0] == 'www':
                netloc_parts.pop(0)

            # List of known ccTLDs
            cctlds = ['ac', 'ad', 'ae', 'af', 'ag', 'ai', 'al', 'am', 'ao', 'aq', 'ar', 'as', 'at', 'au', 'aw', 'ax', 'az', 'ba', 'bb', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bl', 'bm', 'bn', 'bo', 'bq', 'br', 'bs', 'bt', 'bv', 'bw', 'by', 'bz', 'ca', 'cc', 'cd', 'cf', 'cg', 'ch', 'ci', 'ck', 'cl', 'cm', 'cn', 'co', 'cr', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'de', 'dj', 'dk', 'dm', 'do', 'dz', 'ec', 'ee', 'eg', 'eh', 'er', 'es', 'et', 'eu', 'fi', 'fj', 'fk', 'fm', 'fo', 'fr', 'ga', 'gb', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi', 'gl', 'gm', 'gn', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gw', 'gy', 'hk', 'hm', 'hn', 'hr', 'ht', 'hu', 'id', 'ie', 'il', 'im', 'in', 'io', 'iq', 'ir', 'is', 'it', 'je', 'jm', 'jo', 'jp', 'ke', 'kg', 'kh', 'ki', 'km', 'kn', 'kp', 'kr', 'kw', 'ky', 'kz', 'la', 'lb', 'lc', 'li', 'lk', 'lr', 'ls', 'lt', 'lu', 'lv', 'ly', 'ma', 'mc', 'md', 'me', 'mf', 'mg', 'mh', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz', 'na', 'nc', 'ne', 'nf', 'ng', 'ni', 'nl', 'no', 'np', 'nr', 'nu', 'nz', 'om', 'pa', 'pe', 'pf', 'pg', 'ph', 'pk', 'pl', 'pm', 'pn', 'pr', 'ps', 'pt', 'pw', 'py', 'qa', 're', 'ro', 'rs', 'ru', 'rw', 'sa', 'sb', 'sc', 'sd', 'se', 'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sr', 'ss', 'st', 'su', 'sv', 'sx', 'sy', 'sz', 'tc', 'td', 'tf', 'tg', 'th', 'tj', 'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tr', 'tt', 'tv', 'tw', 'tz', 'ua', 'ug', 'uk', 'us', 'uy', 'uz', 'va', 'vc', 've', 'vg', 'vi', 'vn', 'vu', 'wf', 'ws', 'ye', 'yt', 'za', 'zm', 'zw']

            # Removing the last part if it matches any of the known ccTLDs
            if len(netloc_parts) > 1 and netloc_parts[-1] in cctlds:
                netloc_parts.pop(-1)

            num_subdomains = len(netloc_parts)

            return 1 if num_subdomains <= 1 else 1 if num_subdomains == 2 else -1
        except:
            return -1