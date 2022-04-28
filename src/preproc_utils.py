import nltk
import wordcloud
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def count_labels(df):
    """ prints labels counts (no positive/neutral/negative labels) """

    labels_stats = df[['Positive', 'Neutral', 'Negative']].sum(0)
    print(f'\nLabels\' counts:\n{labels_stats}')
    labels_freq = labels_stats / labels_stats.sum()
    print(f'\nLabels\' frequencies:\n{labels_freq}')

    return None


def add_tokens_column(df):
    """ Returns dataframe with tokens created out of sentencs

    Converts sentence to lower case, remove stopwords and some punctuation,
    create tokens that are held in a list in a new column.

    Args:
        df (DataFrame): DataFrame with sentences in 'Sentence' column, to be tokenized.

    Returns:
        df (DataFrame): DataFrame with 'Tokens' column.

    """
    stopwords_eng = set(nltk.corpus.stopwords.words('English'))

    # df['Tokens'] = df['Sentence'].str.lower()\
    #     .apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords_eng])) \
    #     .apply(lambda x: nltk.word_tokenize(x))

    tokens = df.Sentence.map(nltk.word_tokenize).rename('Tokens')
    df_tokens = pd.concat([df, tokens], axis=1)
    punctuation = ['.', ',', '(', ')', '-', '%', ':', '+', '-', '"', '\'']
    df_tokens['Tokens'] = df_tokens['Tokens'].apply(lambda x: [
        y.lower() for y in x if y.lower() not in stopwords_eng and y.lower() not in punctuation])

    df_tokens['Tokens'] = df_tokens['Tokens'].apply(lambda x: [re.sub(r'[^A-Za-z ]+', '', y) for y in x])

    return df_tokens

def token_lemmatizer(df):
    """ Returns dataframe with tokens lemmatized

        Takes dataframe with Tokens column that contains tokens and convert them using lemmatizer

        Args:
        df (DataFrame): DataFrame with tokens in Token column.

        Returns:
        df (DataFrame): DataFrame with 'Tokens' column after lemmatization.

"""

    lemmatizer = nltk.stem.WordNetLemmatizer()
    result = df.Tokens.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return result



def lowercase_in_sentence(df):
    """ makes words in Sentence column lowercase """
    df['Sentence'] = df['Sentence'].str.lower()
    return df


def get_negatives():
    """ returns list with negative words

    source of the list: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    via https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107

    """
    negative_file_path = "../data/negative-words.txt"
    with open(negative_file_path) as source_file:
        negative_words = [i.replace("\n", "") for i in source_file if not (i.startswith(';') or i == "\n")]

    return negative_words


def get_positives():
    """ returns list with positive words

    source of the list: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    via https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107

    """
    positive_file_path = "../data/positive-words.txt"
    with open(positive_file_path) as source_file:
        positive_words = [i.replace("\n", "") for i in source_file if not (i.startswith(';') or i == "\n")]

    return positive_words


def cosine_similarity_check(df, threshold=0.9):
    """ Check for sentences with high cosine similarity

    Calculate cosine similarity pair-wise and return sentences idx, check only pairs which similarity is above
    threshold, get an id of one of those sentences, return list with ids.

    Args:
        df (DataFrame): DataFrame with sentences in 'Sentence' column for similarity check.
        threshold (float): Only similarities above given treshold will be considered.

    Returns:
        df (DataFrame): DataFrame with 'Tokens' column.

    """

    corpus = list(df["Sentence"].values)
    vectorizer = TfidfVectorizer()
    vectorized_corpus = vectorizer.fit_transform(corpus)

    to_remove = []
    for x in range(0, vectorized_corpus.shape[0]):
        for y in range(x, vectorized_corpus.shape[0]):
            if x != y:
                if cosine_similarity(vectorized_corpus[x], vectorized_corpus[y]) > threshold:
                    # print(f"{corpus[x]} | ID: {x}")
                    # print(f"{corpus[y]} | ID: {y}")
                    # print("Cosine similarity:", cosine_similarity(vectorized_corpus[x], vectorized_corpus[y]))
                    to_remove.append(y)
    return list(set(to_remove))

def split_sentences(df):
    positive_sentences = df[df.Positive == 1]
    negative_sentences = df[df.Negative == 1]
    neutral_sentences = df[df.Neutral == 1]

    return positive_sentences, negative_sentences, neutral_sentences

def stopwords_ratio(sentence, stopwords):
    """ Calculates a ratio of stopwords in a sentence

    Args:
        sentence (str): Sentence to analyze.
        stopwords (set): Set with stopwords.

    Returns:
        (float): Stopwords ratio.

    """

    num_total_words = 0
    num_stopwords = 0
    for token in nltk.word_tokenize(sentence):
        if token in stopwords:
            num_stopwords += 1
        num_total_words += 1
    return num_stopwords/num_total_words

def mean_token_length(sentence):
    """ Calculates an average token length

    Args:
        sentence (str): Sentence to analyze.

    Returns:
        (float): An average token length

    """

    token_lengths = []
    for token in nltk.word_tokenize(sentence):
        token_lengths.append(len(token))

    return np.array(token_lengths).mean()

def make_wordcloud(analyzed_string, figure_filename):
    """ Prepare wordcloud plot

    Args:
        sentence (str): String with words to analyze.
        figure_filename (str): Destination for a png file with wordcloud plot.

    Returns:
        (float): An average token length
    """

    wordcloud_plot = wordcloud.WordCloud(width=1200, height=800, background_color='white', min_font_size=10).generate(
        analyzed_string)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud_plot)
    plt.axis('off')
    # plt.tight_layout(pad = 0)
    plt.show()
    plt.savefig(figure_filename)

    return None

def add_label(df):
    """ returns dataframe with one column for label"""

    labels = df[['Positive', 'Negative', 'Neutral']].idxmax(1).to_list()
    df.insert(2, "Label", labels, True)

    return df
