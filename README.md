#### Warning: NSFW language in code files / data set

# Introduction
In 2017, seven out of 10 Singaporeans are active on social media platforms on mobile devices - more than double the global average (http://www.businesstimes.com.sg/consumer/7-in-10-singaporeans-use-social-media-on-mobile-double-global-average-survey). With these platforms, Singaporeans are able to leave digital footprints in the form of online comments and social media posts. However, we are all too familiar with situations where people use the Internet to perform criminal deeds. For example, terrorist groups have long used Twitter to radicalise followers; some Singaporeans have also left racially discriminatory remarks on Facebook. It is then natural for us to ask, how can a person of authority monitor a digital city filled with a mix of innocent and _unsafe_ information?

A possible approach is to create Machine Learning models that are able to digitally profile an online user or a website. For example, certain features of an individual or website can be used to judge whether one should flag them for further evaluation. It is likely that such models consider features such as how many _unsafe_ comments had a person posted or what kind of facebook pages he  follows. But how does one characterise comments? As such, our group proposes a novel Machine Learning model that is able to classify whether a standalone comment is _unsafe_.

# Methodology
We will use a common corpus-vectorization method to decompose texts into numerical data. However, we will adopt a modified version of the model proposed in (*Wang,  S.,  and  Manning,  C.  D.2012.  Baselines and bigrams: Simple, good sentiment and topic  classification*) to produce a logistic regression model built over NB log-count ratios as features values. This modified version has the following highlights:
- Using _Term Frequency - Inverse Document Frequency_ values for Naive Bayes log-count ratio instead of binary count of words. This reduces the influence of common innoculous words/ngrams/character-grams on predictions and focuses more on 'rare' and influential words/ngrams/character-grams.
- Using NB ratios on _tf-idf_ values to capture 'bad' words which occur fairly frequently in _unsafe_ comments but not frequent enough to be reflected in their _tf-idf_ values.
- Emoji stemming - we convert emoji into words

 # Overview of results
We then compare the results of this model (A linear model after feature engineering) with other conventional models such as LSTM with a metric such as AUC score.
 ![alt text](/results/results.png)
 
