# -*- coding: utf-8 -*-

import warnings
import pickler as pick
import tweet_catch as tC
import spam_classifier as sC
import matplotlib.pyplot as plt

vectorizer, mnb = pick.loadMNB()
warnings.filterwarnings('ignore')

print('*********** DATASET ***********')
print(sC.tweet_preview)
print('\n------------------\n')     
print('*********** DATASET GROUPED BY LABEL ***********')
print(sC.tweet_group)
print('\n------------------\n')

plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(sC.spam_Vcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
print('*********** SPAM WORD CLOUD ***********')
plt.show()
print('\n------------------\n')

plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(sC.ham_Vcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
print('*********** HAM WORD CLOUD ***********')
plt.show()
print('\n------------------\n')

print('*********** SHOWCASE OF TRAINING DATA ***********')
print(sC.train_preview)
print('\n')

print('*********** TRAINING DATA GROUPED BY LABEL ***********')
print(sC.train_group)
print('\n------------------\n')


if __name__ == '__main__':

    print("\n*********** TEST PHASE WITH LIVE TWEET ***********\n")
    
    twitter_client = tC.TwitterClient()
    tweets = twitter_client.get_live_feed(1)
    tweetList = []
    tweetList.append(tweets[0].text)
    accused_user = tweets[0].user.screen_name
    
    input_transformed = vectorizer.transform(tweetList)
    prediction = mnb.predict(input_transformed)
    
    print('Analyzed live-tweet:', tweetList)
    print('By user: @'+ accused_user)
    print('\nAccording to MNB Classification this tweet is', 'SPAM' if prediction else 'HAM')