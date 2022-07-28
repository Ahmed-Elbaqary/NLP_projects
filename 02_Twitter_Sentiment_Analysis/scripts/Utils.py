import numpy as np
import re
import string
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Process tweet function.
        Input:
            tweet: a string containing a tweet
        Output:
            tweets_clean: a list of words containing the processed tweet
    """
    STOCK_MARKETS = re.compile(r"\$\w*")
    OLD_TWEET_STYLE = re.compile(r"^RT[\s]+")
    HYPERLINKS = re.compile(r"https?:\/\/.*[\r\n]*")
    HASH_SIGN = re.compile(r"#")

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

    # Remove all unwanted strings
    tweet = STOCK_MARKETS.sub("", tweet)
    tweet = OLD_TWEET_STYLE.sub("", tweet)
    tweet = HYPERLINKS.sub("", tweet)
    tweet = HASH_SIGN.sub("", tweet)

    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, y):
    """Build frequencies.
        Input:
            tweets: a list of tweets
            ys: an m x 1 array with the sentiment label of each tweet
                (either 0 or 1)
        Output:
            freqs: a dictionary mapping each (word, sentiment) pair to its
                frequency
    """
    y_list = np.squeeze(y).tolist()

    freqs = Counter()
    for y, tweet in zip(y_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] += 1

    return freqs