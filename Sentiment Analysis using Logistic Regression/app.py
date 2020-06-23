import streamlit as st
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
import nltk
import os
import pickle

from utils import *

theta = np.zeros((3,1))
theta[0,0] = 0
theta[1,0] = 0.0009
theta[2,0] = -0.0008

@st.cache(persist=True)
def load_data():
    nltk.download('twitter_samples')
    nltk.download('stopwords')

    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    X = all_positive_tweets + all_negative_tweets
    y = np.append(np.ones((len(all_positive_tweets), 1)), np.zeros((len(all_negative_tweets), 1)), axis=0)
    return X, y


@st.cache(persist=True)
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


@st.cache(persist=True)
def build_features(X, freqs):
    features = np.zeros((len(X), 3))
    for i in range(len(X)):
        features[i, :] = extract_features(X[i], freqs)
    return features


def main():
    st.title("Twitter Sentiment Analysis")

    X, y = load_data()
    p = np.random.randint(0, len(X))
    if st.button("Generate Random Tweet"):
        if(int(y[p]) == 1):
            st.markdown('<p style="color:green"> {} </p>'.format(str(X[p])),unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:red"> {} </p>'.format(str(X[p])),unsafe_allow_html=True)

    
    st.markdown("## Preprocessing Phase")
    st.write("This phase let you know how your tweet will be processed. I have preprocessed your tweet so that it does not contain any stopword, punctuations, hashtags and URLs")
    st.write("Enter your tweet below to know how it will be processed")
    text = st.text_input("Enter your Tweer here", key=1)
    if(text != ""):
        st.markdown("### Preprocessed Text:")
        st.markdown('<i> {} </i>'.format(' '.join(process_tweet(text))), unsafe_allow_html=True)
    else:
        st.warning("It cannot be left blank.")

    
    st.markdown("## Classification Phase")
    st.write("In this phase we'll determine whether your Tweet is Positive or Negative")
    
    with open('dictionary.pickle', 'rb') as handle:
        freqs = pickle.load(handle)

    text = st.text_input("Enter your Tweer here", key=2)
    if(text != ""):
        f = extract_features(text, freqs)
        y_pred = sigmoid(np.dot(f, theta))
        if(y_pred >= 0.5):
            st.markdown("<center> <h1 style='color:green' style='font-size:30px'> POSITIVE </h1> </center>", unsafe_allow_html=True)
        else:
            st.markdown("<center> <h1 style='color:RED' style='font-size:30px'> NEGATIVE </h1> </center>", unsafe_allow_html=True)
    else:
        st.warning("It cannot be left blank.")

if __name__ == "__main__":
    main()