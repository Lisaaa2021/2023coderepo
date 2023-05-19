import sklearn
import csv
import gensim
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

feature_to_index = {
    'token': 1,
    'pos': 2,
    'chunk_tag': 3,
    'token_right': 5,
    'token_left': 6,
    'cap_type': 7
}


def extract_word_embedding(token, word_embedding_model):
    '''
    Function that returns the word embedding for a given token out of a distributional semantic model and a 300-dimension vector of 0s otherwise

    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0] * 300
    return vector


def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row

    :param row: row from conll file
    :param selected_features: list of selected features
    :type row: string
    :type selected_features: list of strings

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values[feature_name] = row[r_index]

    return feature_values


def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values

    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)

    :returns: vectorizer with feature values fitted
    '''
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)

    return vectorizer


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


def extract_traditional_features_and_embeddings_plus_gold_labels(
        conllfile, word_embedding_model, vectorizer=None):
    '''
    Function that extracts traditional features as well as embeddings and gold labels using word embeddings for current and preceding token

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    dense_vectors = []
    traditional_features = []

    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter=',', quotechar='|')
    for index, row in enumerate(csvreader):
        if index == 0:
            continue
        if len(row) == 8:
            token_vector = extract_word_embedding(row[1], word_embedding_model)
            pt_vector = extract_word_embedding(row[6], word_embedding_model)
            dense_vectors.append(np.concatenate((token_vector, pt_vector)))
            #mixing very sparse representations (for one-hot tokens) and dense representations is a bad idea
            #we thus only use other features with limited values
            other_features = extract_feature_values(row, ['cap_type'])
            traditional_features.append(other_features)
            #adding gold label to labels
            labels.append(row[4])

    #create vector representation of traditional features
    if vectorizer is None:
        #creates vectorizer that provides mapping (only if not created earlier)
        vectorizer = create_vectorizer_traditional_features(
            traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(
        dense_vectors, sparse_features)

    return combined_vectors, vectorizer, labels


def label_data_with_combined_features(testfile, classifier, vectorizer,
                                      word_embedding_model):
    '''
    Function that labels data with model using both sparse and dense features
    '''
    feature_vectors, vectorizer, goldlabels = extract_traditional_features_and_embeddings_plus_gold_labels(
        testfile, word_embedding_model, vectorizer)
    predictions = classifier.predict(feature_vectors)

    return predictions, goldlabels


def create_classifier(features, labels):
    '''
    Function that creates classifier from features represented as vectors and gold labels

    :param features: list of vector representations of tokens
    :param labels: list of gold labels
    :type features: list of vectors
    :type labels: list of strings

    :returns trained logistic regression classifier
    '''

    lr_classifier = LogisticRegression(solver='saga')
    lr_classifier.fit(features, labels)

    return lr_classifier


def create_svc_classifier(features, labels):
    '''
    Function that creates classifier from features represented as vectors and gold labels

    :param features: list of vector representations of tokens
    :param labels: list of gold labels
    :type features: list of vectors
    :type labels: list of strings

    :returns trained logistic regression classifier
    '''

    model = LinearSVC(C = 1)
    model.fit(features, labels)

    return model
