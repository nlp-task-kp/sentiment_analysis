import nltk
import sklearn.metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Vader_result(sentence):
    """ Calculate simple, rule-based VADER score"""

    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    if sia.polarity_scores(sentence)["compound"] > 0.05:
        result = 'Positive'
    elif sia.polarity_scores(sentence)["compound"] > -0.05:
        result = 'Neutral'
    else:
        result = 'Negative'

    return result

# Simple VADER Sentiment Analysis ----

def vader_sentiment_analysis(sentences):
    sentences_extended = sentences
    labels = sentences_extended[['Positive', 'Negative', 'Neutral']].idxmax(1).to_list()
    sentences_extended.insert(2, "Label", labels)

    correct = 0
    vader_predictions = []
    for _, row in sentences_extended.iterrows():
        prediction = Vader_result(row['Sentence'])
        vader_predictions.append(prediction)
        if prediction == str(row['Label']):
            correct += 1
    accuracy = sklearn.metrics.accuracy_score(vader_predictions, sentences_extended['Label'])
    confusion_matrix = sklearn.metrics.confusion_matrix(sentences_extended['Label'], vader_predictions)
    print(f"Accuracy for the VADER senitment analysis: {accuracy}")
    print(f"Confusion matrix for the VADER senitment analysis: {confusion_matrix}")
    sentences_extended.drop("Label", 1, inplace=True)
    return None


