import warnings
import toml
import logging

import nltk
import nltk.sentiment

import pandas as pd
import numpy as np

from src import preproc_utils
from src import ml_utils

def main():
    """ main code"""

    warnings.simplefilter(action='ignore', category=FutureWarning)
    # in production should be controller, here - for not cluttering the output

    downloaded_sentences = data_import("data/sentences_with_sentiment.xlsx") #TODO: could be moved to config
    sentences_cleaned = data_preprocessing(downloaded_sentences)
    data_analysis(sentences_cleaned)

    return None

def data_preprocessing(sentences):
    """ preprocessing imported dataset """

    print(f'Preprocessing dataset - lowering case and dropping duplicates')
    sentences_lowercase = preproc_utils.lowercase_in_sentence(sentences)
    no_duplicates = sentences_lowercase.drop_duplicates(subset=['Sentence', 'Positive', 'Negative', 'Neutral'])
    # log no duplicates dropped

    print(f'Preprocessing dataset - dropping very similar sentences')
    no_duplicates.reset_index(drop=True, inplace=True)
    indexes_list = preproc_utils.cosine_similarity_check(no_duplicates, threshold=0.99)  # fine when dataset is small
    no_similar_sentences = no_duplicates.drop(indexes_list)
    no_similar_sentences.shape
    # log no rows dropped

    return no_similar_sentences

def data_analysis(sentences):
    ml_utils.vader_sentiment_analysis(sentences)
    # very low accuracy of 0.54, to be expacted in case of a rule-based method, domain-specific lang.

    ml_utils.vader_sentiment_analysis(sentences)
    # very low accuracy of 0.64

    return None

def data_import(input_path):
    """ importing dataset with EPARs """

    print(f'Importing dataset from: {input_path}')
    imported_sentences = pd.read_excel(input_path)
    print(f'Imported dataset dimensions: {imported_sentences.shape}')

    # couple of simple sanity checks, should be modularized into sanity_checks()
    if imported_sentences.empty:
        raise ValueError('Imported data frame is empty')

    if any(imported_sentences[['Positive', 'Neutral', 'Negative']].sum(1) != 1):
        warnings.warn("every sentence should have only one label.", UserWarning) #TODO: this could be handled

    return imported_sentences


if __name__ == '__main__':
    main()

