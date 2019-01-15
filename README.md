# Project Report #
1. Introduction
2. Methodology (Introduce 3 models, what they are about)
3. Data
4. Performance
5. Other results

# Things to do next: #
- Discuss the methodology that an expert can use to make our predicting model useful (Example: Our model's goal is to classify different text type. What an expert can do is to create a model to predict user profile based on how many toxic comments, frequency of toxic comments he post online - the expert can use our model as an intermediate step to solely check if a comment is toxic or not.) This is mainly done to address Prof Bryan's concern in his email regarding how can we actually use the results for useful things.
**(Use our model as part of a bigger prediction models -e.g government has a huge model to detect whether a facebook page should be flagged for inspection; this model uses features such as whether the page posts toxic contents, how many bad comments are posted everyday etc. But to obtain these features, the government can use our localized model to judge whether a comment is toxic in the first place; errors in our prediction will be passed on as noise to the bigger model and the bigger model can handle these noise. Or the owner can use our model to monitor activities on his website)**
- Divide the test set/train set better. I feel that the training data set contains too many clean comments; this is a major issue because skewed data can cause inaccuracy in predictions. We need to ensure training data set contains more toxic comments. There is a few ways to do this:
  - Decrease the number of clean comments manually in the training data set so that the number of toxic comments and clean comments is more balanced while training
  - Artificially repeat toxic comments so that we can increase the number of toxic comments in training data set. This means we replicate toxic comments in the training data (so maybe each toxic comment we multiply 5 times in training data). Some experts recommend this, some people online says this is a bad approach. We need more discussions on this.
- Instead of test/train set, We need to instead partition into **training data, development data and test data**. (Sorry I forgot to mention this earlier) So we train using training data, tune parameters while testing our model on development data, and finally present out results using test data which will be the only time the test data will be passed to our model.
- We also have to present the test accuracy on **training data** because this can give us some indication on overfitting. Ideally we want the accuracy of training data to be similar to development data.
- Tune more parameters for word_vectorizer. We can increase the number of word sequences considered (right now is only 1 and 2 word sequences) or other things. Check http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
to see parameteres regarding text decomposition.
- Discuss more models to use! Prof Bryan says we should compare results across different model to point out improvements :)


# toxic2018
https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams

# Feature Engineering
- Break down comments into character/word ngrams
- computer tf-idf score for each word for each comment

# Modelling
- Either use logistic regression or other methods

# Results
- Use AUC-ROC curve to determine accuracy

resources:
https://en.wikipedia.org/wiki/Logistic_regression (for Logistic regression)
https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf (for naive bayes transformation followed by logistic regression procedure)







