import re
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags, only removing the hash symbol
    tweet = re.sub(r'#', '', tweet)
    # tokenize the tweet
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweet_clean.append(stem_word)

    return tweet_clean


def sigmoid(z):
    h = np.divide(1, 1 + np.exp(-z))
    return h

def gradientDescent(X, y, theta, alpha, num_iters):
    m = X.shape[0]

    for i in range(0, num_iters):
        z = np.dot(X, theta)
        h = sigmoid(z)
        J = -(np.dot(y.T, np.log(h)) + np.dot((1 - y.T), np.log(1 - h))) / m
        theta = theta - alpha * np.dot(X.T, h - y) / m

    J = float(J)
    return J, theta


def extract_features(tweet, freqs):
    processed_tweet = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0,0] = 1

    for word in processed_tweet:
        x[0, 1] += freqs.get((word, 1), 0)
        x[0, 2] += freqs.get((word, 0), 0)
    
    return x

