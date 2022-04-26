import nltk

import pandas as pd
import numpy as np

def count_labels(df):
    """ prints labels counts (no positive/neutral/negative labels) """

    labels_stats = df[['Positive', 'Neutral', 'Negative']].sum(0)
    print(f'\nLabels\' counts:\n{labels_stats}')
    labels_freq = labels_stats/labels_stats.sum()
    print(f'\nLabels\' frequencies:\n{labels_freq}')

    return None

def add_tokens_column(df):
    """ Returns dataframe with tokens created out of sentencs

    Converts sentence to lower case, remove stopwords, create tokens that are held in a list in a new column.

    Args:
        df (DataFrame): DataFrame with sentences in 'Sentence' column, to be tokenized.

    Returns:
        df (DataFrame): DataFrame with 'Tokens' column.

    """
    stopwords_eng = nltk.corpus.stopwords.words('English')
    df['Tokens'] = df['Sentence'].str.lower()\
        .apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords_eng])) \
        .apply(lambda x: nltk.word_tokenize(x))

    return df

def lowercase_in_sentence(df):
    """ makes words in Sentence column lowercase """
    df['Sentence'] = df['Sentence'].str.lower()
    return df



