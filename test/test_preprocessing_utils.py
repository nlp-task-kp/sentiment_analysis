import pytest

import pandas as pd

import preprocessing_utils

@pytest.fixture
def input_dataframe():
    sentences = {'Sentence': ["Hello World!", "Hello world!", "Hello agaiN", "Bad News", "It's April."],
         'Positive': [1, 1, 1, 0, 0],
         'Neutral': [0, 0, 0, 0, 1],
         'Negative': [0, 0, 0, 1, 0]}
    df = pd.DataFrame(data=sentences)

    return df

def test_add_tokens_column(input_dataframe):
    assert preprocessing_utils.add_tokens_column(input_dataframe)['Tokens'][2] == ["hello"]
    assert preprocessing_utils.add_tokens_column(input_dataframe)['Tokens'][3] == ["bad", "news"]

def lowercase_in_sentence(input_dataframe):
    assert preprocessing_utils.lowercase_in_sentence(input_dataframe)['Sentence'][4] == ["it's april."]