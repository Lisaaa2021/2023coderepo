import pandas as pd
from gensim.models import KeyedVectors
import numpy as np

#google_news = 'data/GoogleNews-vectors-negative300.bin'
#wv_from_bin = KeyedVectors.load_word2vec_format(google_news, binary=True)
#word2vec binary filepath

############### Add features#################

def extract_left_token(token_series):
    token_series_duplicate = token_series
    token_left = ['.']
    for ind, token in enumerate(token_series):
        if ind > 0:
            token_left.append(token_series_duplicate[ind - 1])
    return token_left


def extract_right_token(token_series):
    token_series_duplicate = token_series
    token_right = []
    series_length = len(token_series)
    for ind, token in enumerate(token_series):
        if ind != series_length - 1:
            token_right.append(token_series_duplicate[ind + 1])
    token_right.append('.')
    return token_right




def cap_type(token_series):
    cap_type = []
    for token in token_series:
        token = str(token)
        if token.isnumeric():
            cap_type.append('isnumeric')
        elif token.islower():
            cap_type.append('islower')
        elif token.isupper():
            cap_type.append('isupper')
        elif token.istitle():
            cap_type.append('istitle')
        else:
            cap_type.append('other')
    return cap_type



###############
def word2vec(word):
    if word in wv_from_bin:
        result = wv_from_bin[word]
    else:
        result = [0] * 300
    return result


def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation

    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists

    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''

    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())

    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector, dense_vectors[index]))
        combined_vectors.append(combined_vector)
    return combined_vectors
