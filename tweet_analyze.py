# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:33:37 2018

@author: bugur
"""

import spam_classifier as SC
import tweet_catch as tC

twitter_client = tC.TwitterClient()
tweets = twitter_client.get_live_timeline(1)
tweetList = []
tweetList.append(tweets[0].text)

input_transformed = SC.vectorizer.transform(tweetList)
prediction = SC.mnb.predict(input_transformed)

print('Analyzed live-tweet:', tweetList)
print('\nAccording to MNB Classification this tweet is', 'SPAM' if prediction else 'HAM')