# Twitter Spam Filter Using Multinomial Naive Bayes Classification with TF-IDF tokenization and Snowball Text Processing

This projects main goal is to apply Multinomial Naive Bayes Classification to a labelled dataset after proccession of texts and tokenization. 

Main workflow of the project is as follows;

1. Loading the dataset
2. Preparing data for processing
3. Processing data(**Snowballing**)
  * Separate the sentence into individual words
  * Convert all letters to lowercase
  * Remove stopwords
  * Don't care for non-English words(depending on your dataset)
4. Vectorize each word to count occurences in the given sentence/data using [TF-IDF](http://www.tfidf.com/)
5. Visualise spam and ham data using word clouds with occurence weights.
6. Scramble dataset and separate data for training and testing purposes
7. Train the algorithm
8. Test the trained model with your test data to obtain accuracy and F1 scores using [Multinomial Naive Bayes](https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes)
9. Save the the trained model (see **model.pkl**)
10. Twitter Authentication and tweet object catching functionalities using [Tweepy](https://tweepy.readthedocs.io/en/3.7.0/)
11. Using personel api endpoint/keys, accessing live feed and extracting the very last tweet
12. Load the trained model
13. Feed the extracted tweet into the model and predict Spamminess/Hamminess

**Do not forget, Bayes Classification is based on probability. If you want more accurate results, use a bigger dataset. I personally mined the dataset that I have and because of [Twitter ToS](https://developer.twitter.com/en/developer-terms/agreement) can not share any plain-text content publicly**

### To run the Project
- Provide your app credentials acquired through [Twitter Developer page](https://developer.twitter.com/en.html) inside twitter_credentials.py with respected values.
- Provide .csv formatted dataset in spam_classifier.py where expected
- If you're using an IDE just run the script; if you're using a console environment while in project folder run
```
python -tt spam_classifier.py
```

# Possible refactoring and commenting is on it's way.

#### I would like to thank my beloved teacher Dr. M. Ali Aydın for giving me this project which pushed me to my limits and made me explore beyond my comfort zone. This project was done as a thesis project for my CE degree.

For any further info or question feel free to contact me.
