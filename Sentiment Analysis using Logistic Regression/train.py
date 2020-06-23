import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
import nltk
import os
import pickle

from utils import *

def load_data():
    nltk.download('twitter_samples')
    nltk.download('stopwords')

    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    X = all_positive_tweets + all_negative_tweets
    y = np.append(np.ones((len(all_positive_tweets), 1)), np.zeros((len(all_negative_tweets), 1)), axis=0)
    return X, y

def build_freqs(tweets, ys):
    """
    tweets: list of tweet
    ys: m x 1 array with sentiment label of each tweet
    """
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, int(y))
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def build_features(X, freqs):
    features = np.zeros((len(X), 3))
    for i in range(len(X)):
        features[i, :] = extract_features(X[i], freqs)
    return features


def main():

    X, y = load_data()

    freqs = build_freqs(X, y)

    features = build_features(X, freqs)

    J, theta = gradientDescent(features, y, np.zeros((3, 1)), 1e-3, 1500)
    print("J: ", J)
    print("Parameters: ", theta)


if __name__ == '__main__':
    main()