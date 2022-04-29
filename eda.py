""" Exploratory Data Analysis """

import nltk.sentiment
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import main #TODO: change!
from src import preproc_utils

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('vader_lexicon')

# Import dataset, check labels and review sentences ----
raw_sentences = main.data_import('data/sentences_with_sentiment.xlsx')
preproc_utils.count_labels(raw_sentences)
# with 60% positive, 26% neutral and only 14% negative, dataset is quite inbalanced

if any(raw_sentences[['Positive', 'Neutral', 'Negative']].sum(1) != 1):
    print('some errors in labelling')
else:
    print('labelling is correct')

raw_sentences.isnull().sum()  # no null values

old_width = pd.options.display.max_colwidth
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 8)
print(raw_sentences.Sentence[30:80])

# Checking for duplicates ----
# lowercase
raw_sentences_lower = preproc_utils.lowercase_in_sentence(raw_sentences)

# no duplicates
no_duplicates_full_subset = raw_sentences_lower.drop_duplicates(subset=['Sentence', 'Positive', 'Negative', 'Neutral'])
no_duplicates_sentences_only = raw_sentences_lower.drop_duplicates(subset=['Sentence'])

preproc_utils.count_labels(no_duplicates_full_subset)
preproc_utils.count_labels(no_duplicates_sentences_only)
# there is no sentence duplicate that would have different labels

# remove very similar sentences (cosine_similarity measure)
unique_sentences = no_duplicates_full_subset
unique_sentences.reset_index(drop=True, inplace=True)
lst = preproc_utils.cosine_similarity_check(unique_sentences)
unique_sentences_cleaned = unique_sentences.drop(lst)
unique_sentences_cleaned.shape

preproc_utils.count_labels(raw_sentences)
preproc_utils.count_labels(unique_sentences_cleaned)

# after duplicates removal dataset becomes even smaller (140, 64, 29 for positive, neutral and negative)
# imbalances are very similar to the original ones (60%, 27% and 12%)

# Checking number of tokens distribution ----

positive_sentences, negative_sentences, neutral_sentences = preproc_utils.split_sentences(unique_sentences_cleaned)

# Checking number of tokens distribution ----

warnings.simplefilter(action='ignore', category=FutureWarning)

preproc_utils.draw_no_tokens(positive_sentences, negative_sentences, neutral_sentences)

# Checking average token length distribution ----

preproc_utils.draw_avg_token_len(positive_sentences, negative_sentences, neutral_sentences)

# Taking into account small number of observations for, at least, Neutral and Negative labels
# and shape of the plots, one can draw a conclustion that those distributions does not provide much value


# Wordclouds ----

preproc_utils.make_wordcloud(' '.join(positive_sentences.Sentence.values), 'figures/eda_wordcloud_positive.png')
preproc_utils.make_wordcloud(' '.join(negative_sentences.Sentence.values), 'figures/eda_wordcloud_negative.png')
preproc_utils.make_wordcloud(' '.join(neutral_sentences.Sentence.values), 'figures/eda_wordcloud_neutral.png')

# Most common words ----

unique_sent_tokens = preproc_utils.add_tokens_column(unique_sentences_cleaned)
pos_sentences_tokens, neg_sentences_tokens, neutr_sentences_tokens = preproc_utils.split_sentences(unique_sent_tokens)

pos_tokens = [a for b in pos_sentences_tokens.Tokens.tolist() for a in b]
most_common = nltk.FreqDist(pos_tokens).most_common(20)
print(f'Most common tokens (in positive sentences): {most_common}')

neg_tokens = [a for b in neg_sentences_tokens.Tokens.tolist() for a in b]
most_common = nltk.FreqDist(neg_tokens).most_common(20)
print(f'Most common tokens (in negative sentences): {most_common}')

neutr_tokens = [a for b in neutr_sentences_tokens.Tokens.tolist() for a in b]
most_common = nltk.FreqDist(neutr_tokens).most_common(20)
print(f'Most common tokens (in neutral sentences): {most_common}')

# there are no visible, significant differences between most common words in positive, negative and neutral sentences
# tokens should be further cleaned

# Check negative words in positive sentences ----

processed_sentences = preproc_utils.add_tokens_column(unique_sentences_cleaned)
negative_words = preproc_utils.get_negatives()

processed_sentences['Tokens_negative'] = processed_sentences['Tokens'].apply(
    lambda tokens_list: [word for word in tokens_list if word in negative_words])
result = processed_sentences[processed_sentences['Tokens_negative'].map(lambda d: len(d)) > 0]
result = result[result['Positive'] == 1]
result = result.sort_values(by=['Tokens_negative'])[['Sentence', 'Tokens', 'Tokens_negative']]
print(f'Positive sentences with negative words, first couple of records: ')
print(result[['Sentence', 'Tokens_negative']][0:10])
print(f'{result.shape}')

# a lot of negative words use in positive sentences, are used with negation, so in context those are not really
# negative words any more (ie. for word 'concern': no safety concern)

# Check number of words in corpus, after cleaning----

unique_sent_tokens = preproc_utils.add_tokens_column(unique_sentences_cleaned)
all_tokens = [a for b in unique_sent_tokens.Tokens.tolist() for a in b]
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(x) for x in all_tokens]
porter = nltk.stem.PorterStemmer()
stemmed_words = [porter.stem(x) for x in lemmatized_words]
print(f"Count of unique words {len(list(set(stemmed_words)))}")
print("after removing stopwords, non alpha characters, lowering case, lemmatizing, stemming")

df = pd.DataFrame(list(set(stemmed_words)))
df.columns = ["words"]
df.sort_values(by="words", inplace=True)

# Count of unique words 971
# one can notice a lot of domain-specific terminology

# Average count of words in a sentence, after tokenization ----

print(unique_sent_tokens.Tokens.apply(lambda x: len(x)).mean())


# Average count of words in a sentence, after tokenization ----
labels = unique_sentences_cleaned[['Positive', 'Negative', 'Neutral']].idxmax(1).to_list()
sns.countplot(x=labels)
plt.xlabel('Label')
plt.show()

preproc_utils.count_labels(unique_sentences_cleaned)  # to compare
