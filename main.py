import warnings
import toml
import logging

import pandas as pd
import numpy as np

#logger = logging.getLogger(__name__)

def main():
    data_import("data/sentences_with_sentiment.xlsx") #TODO: move to config
    return None

def data_import(input_path):
    """
    importing dataset with EPARs
    :param input_path: path to dataset
    """

    print(f'Importing dataset from: {input_path}') #TODO: change to logging
    imported_sentences = pd.read_excel(input_path)
    print(f'Imported dataset dimensions: {imported_sentences.shape}')

    # couple of simple sanity checks
    if imported_sentences.empty:
        raise ValueError('Imported data frame is empty')

    if any(imported_sentences[['Positive', 'Neutral', 'Negative']].sum(1) != 1):
        warnings.warn("every sentence should have only one label.", UserWarning) #TODO: this should be handled

    return imported_sentences

if __name__ == '__main__':
    main()

