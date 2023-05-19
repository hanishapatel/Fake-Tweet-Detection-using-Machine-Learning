from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import re
import string
from scipy.sparse import hstack

import os
import tweepy as tw

app = Flask(__name__)
pickle_in = open('model_fakenews.pickle','rb')
classifier = pickle.load(pickle_in)
tfid = open('tfidf.pickle','rb')
vectorization = pickle.load(tfid)


consumer_key= 'vPBSIb38pXD74EKJvULzBGbvg'
consumer_secret= 'MT1sePKda0gFh0Nb9NJTF1rRPeEQlU5xmcEuhbBbqwHB52pz9F'
access_token= '3188894918-mYJFrZoUi716btRLz17y5mzAYKgIrsy2m25tH1w'
access_token_secret= 'Bfw06ypwxtDqRCacIVwKMZTYRszDmq0qlRLmjC46AlwWu'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True, parser=tw.parsers.JSONParser())




def wordopt(text):    
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


@app.route('/')
def home():
        return render_template("dashboard.php")

import json
@app.route('/newscheck', methods=['GET','POST'])
def newscheck():
        proceed = 1
        result = -1
        news = ''
        result = ''
        
        tweet_id = request.form['tweet_id']
        
    
        try:
                tweet = api.get_status(str(tweet_id))
        except:
                result = -1
                proceed = 0
                
        
        if proceed == 1:
                news = tweet['text']
                favorite_count = tweet['favorite_count']
                retweet_count = tweet['retweet_count']
                followers_count = tweet['user']['followers_count']
                friends_count = tweet['user']['friends_count']

                
            
                #testing_news = {"text":[news]}
                news_test = pd.DataFrame({"text":[news]})
                
                news_test["text"] = news_test["text"].apply(wordopt)

                
                news_test["favourites_count"] = favorite_count
                news_test["retweet_count"] = retweet_count
                news_test["followers_count"] = followers_count
                news_test["friends_count"] = friends_count

                
                #applying tfidf vectorizer
                news_text_data = vectorization.transform(news_test['text'].values)

                
                import scipy
                news_meta_data = news_test.drop(['text'],axis = 1)
                
                dummy_news_meta = scipy.sparse.csr_matrix(news_meta_data.values)
                final_news_data = hstack((news_text_data, dummy_news_meta)) 
                
                result = classifier.predict(final_news_data)
                
                

        return render_template('dashboard.php', data2 = result , text = news)
        
                









# def test_news(text,favourites_count,retweet_count,followers_count,friends_count,vectorizer,classifier):
#     testing_news = {"text":[news]}
#     news_test = pd.DataFrame(testing_news)

#     news_test["text"] = news_test["text"].apply(wordopt)
#     # new_x_test = [new_def_test["text"],1,120,750,500] 
#     # test = news_test["text"]
    
#     news_test["favourites_count"] = favourites_count
#     news_test["retweet_count"] = retweet_count
#     news_test["followers_count"] = followers_count
#     news_test["friends_count"] = friends_count
    
#     #applying tfidf vectorizer
#     news_text_data = vectorization.transform(news_test['text'].values)
    
#     import scipy
#     news_meta_data = news_test.drop(['text'],axis = 1)
#     dummy_news_meta = scipy.sparse.csr_matrix(news_meta_data.values)
#     final_news_data = hstack((news_text_data, dummy_news_meta))
    
#     result = classifier.predict(final_news_data)
    
#     if result == 0:
#         res = 'FALSE'
#     if result == 1:
#         res = 'TRUE'
    
#     return res

if __name__ == "__main__":
	app.run(debug = True)



