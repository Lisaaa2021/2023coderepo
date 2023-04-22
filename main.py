from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import classification_report
import csv
import numpy as np
import sys


def load_data(openfile):
    # Get a list of list of list (from Angel)
    with open(openfile, encoding='utf-8') as my_file:
        sentence = []
        sentences = []
        for line in my_file:
            row = line.strip('\n').split('\t')

            # if a sentence finishes:
            if len(line.strip('\n')) == 0:
                sentences.append(sentence)  # here is a new sentence
                sentence = []

            elif line.startswith('#'):
                pass

            else:
                sentence.append(row)

    return sentences


def split_predicates(openfile):
    '''
  Splits a sentence based on its predicates.
  Identifies the predicates in a sentence and duplicates it for each predicate with its corresponding argument column.

  Returns: a list of sentences split based on their predicats and arguments
  '''
    full_text = load_data(openfile)  #code requires full text from load data to be in three layer list format (sentences, rows, columns)
    split_list = []

    for sentence in full_text:

        predicates = []
        predicates_pos1 = []
        predicates_pos2 = []
        for row in sentence:
            if row[10] != "_" and row[10] != '':
                predicates.append(row[2])
                predicates_pos1.append(row[3])
                predicates_pos2.append(row[4])
                #labels of the first predicate

        for ind, (predicate, predicate_pos1, predicate_pos2) in enumerate(
                zip(predicates, predicates_pos1, predicates_pos2)):

            dup_sentences = []
            for row in sentence:

                if len(row) == len(predicates) + 11 and str(row[6]).isdigit():
                    split_row = row[:
                                    11]  #for each row in sentence, takes the first 11 columns
                    # append predicate information
                    split_row[10] = predicate
                    split_row.append(predicate_pos1)
                    split_row.append(predicate_pos2)
                    split_row.append(
                        row[10 + (ind + 1)]
                    )  #then appends the corresponding arguments column (note 'num in range(count)' starts at 0, hence the '+1' here)

                    dup_sentences.append(
                        split_row)  #append all rows of sentence to a list

            split_list.append(
                dup_sentences)  #append sentence to the full split_list

    return split_list


def feature_and_gold_extraction(sentences):
    """
  Extracts features in a dictionary and saves gold labels from list of list of lists of data
  returns features: list of dictionaries containing features per token
  returns gold_labels: list of gold labels per token
  """
    features = []
    gold_labels = []

    head_numbers = []
    for ind, sentence in enumerate(sentences):
        for token in sentence:
            feature_dict = {}
            feature_dict["sentence_position"] = token[0]
            feature_dict["raw_token"] = token[1]
            feature_dict["lemma"] = token[2]
            feature_dict["pos"] = token[3]
            feature_dict["universal_pos"] = token[4]
            feature_dict['predicate'] = token[10]
            feature_dict['predicate_pos1'] = token[11]
            feature_dict['predicate_pos2'] = token[12]

            # full list
            # feature_list = ['Definite', 'PronType', 'Number', 'Mood', 'Person', 'Tense', 'VerbForm', 'NumType', 'Degree', 'Case', 'Gender', 'Poss', 'Voice', 'Foreign', 'Reflex', 'Typo', 'Abbr']
            feature_list = [
                'Definite', 'PronType', 'Number', 'Mood', 'Person', 'Tense',
                'VerbForm', 'Case', 'Voice'
            ]

            extra_features = token[5].split("|")
            for feat in feature_list:
                for x in extra_features:
                    feature_dict[feat] = 'X'
                    if x.startswith(feat):
                        split_feature = x.split("=")
                        feature_dict[split_feature[0]] = split_feature[1]

            feature_dict["head_number"] = int(token[6])
            feature_dict["dependency_label"] = token[7]
            feature_dict['extended dependency'] = token[8]
            if str(token[13]) == '_' or str(
                    token[13]
            ) == '':  #cleans data, makes all non-arguments in gold 'O'
                gold = 'O'
            else:
                gold = token[13]  #if using the sentences from split_predicats list, gold will always be index 13

            head_numbers.append((token[1], int(token[6])))
            features.append(feature_dict)
            gold_labels.append(gold)

    return features, gold_labels


def write_features_to_file(sentences, gold_labels, writefile):
    """
  Function to write extracted features and gold labels to file
  """
    df = pd.DataFrame(sentences, columns=['sentence_position','lemma', 'pos',
             'universal_pos', 'predicate', 'predicate_pos1',
             'predicate_pos2'
             'Definite', 'PronType', 'Number', 'Mood', 'Person',
             'Tense', 'VerbForm', 'Case', 'Voice', 'head_number',
             'dependency_label', 'path', 'is_head', 'gold'])
    df["gold"] = np.array(gold_labels)
    df.to_csv(writefile, sep='\t', header=None, quoting=csv.QUOTE_NONE)


def write_predictions_to_file(df, writefile):
    df.to_csv(writefile, sep='\t', header=None, quoting=csv.QUOTE_NONE)

def get_first_label(item):
    item = str(item)
    if item == 'V':
        label = 'predicate'
    elif 'ARG' in item:
        label = 'Yes'
    elif item == 'O':
        label = 'No'
    else:
        label = 'unknown'

    return label


def read_in_preprocessed_data(datafile):
    """
  Reads in file as a pandas dataframe with all columns
  """

    df = pd.read_csv(
        datafile,
        sep='\t',
        header=None,
        #  names=['sentence_position',,'lemma', 'pos',
        #      'universal_pos', 'predicate', 'predicate_pos1',
        #      'predicate_pos2'
        #      'Definite', 'PronType', 'Number', 'Mood', 'Person',
        #      'Tense', 'VerbForm', 'Case', 'Voice', 'head_number',
        #      'dependency_label', 'path', 'is_head', 'gold'],
        quoting=csv.QUOTE_NONE,
        encoding="latin-1")

    df.rename(columns={
        0: 'sentence_position',
        1: 'word',
        2: 'lemma',
        3: 'pos1',
        4: 'pos2',
        5: 'predicate',
        6: 'predicate_pos1',
        7: 'predicate_pos2',
        8: 'Definite',
        9: 'PronType',
        10: 'Number',
        11: 'Mood',
        12: 'Person',
        13: 'Tense',
        14: 'VerbForm',
        15: 'Case',
        16: 'Voice',
        17: 'head_number',
        18: 'basic_dependency',
        19: 'extended_dependency',
        20: 'gold'
    },
              errors="raise",
              inplace=True)

    df['1st_label'] = df['gold'].apply(get_first_label)
    df['2nd_label'] = df['gold']
    df = df[df['1st_label'].isin(['Yes', 'No'])]

    return df

def preprocess(path):
    sentences = split_predicates(path)
    features, labels = feature_and_gold_extraction(sentences)
    outfile = path.replace(".conllu", '') + "_preprocessed.conllu"
    write_features_to_file(features, labels, outfile)

    return outfile

def argument_identifier(train_features, train_targets):
    """ First Logistic Regression model, specifies for each token if it is an argument or not
  returns model: trained LogisticRegression() model
  returns vec: OneHotEncoder()"""
    logreg = LogisticRegression(max_iter=10000)
    vec = OneHotEncoder(handle_unknown='ignore')
    features_vectorized = vec.fit_transform(train_features)
    model = logreg.fit(features_vectorized, train_targets)

    return model, vec


def argument_classifier(train_features, train_targets):
    """ Second Logistic Regression model, takes outputs of argument_identifier model as inputs, gives each argument a label
  returns model: trained LogisticRegression() model
  returns vec: OneHotEncoder()"""
    logreg = LogisticRegression(max_iter=10000)
    vec = OneHotEncoder(handle_unknown='ignore')
    features_vectorized = vec.fit_transform(train_features)
    model = logreg.fit(features_vectorized, train_targets)

    return model, vec


def training_pipeline(datafile, selected_features_1st, selected_features_2nd):
    """The training pipeline of two logstic models for argument identification and argument classification.
    Input: 1.preprocessed training datafile path 2. two selected feature lists for each classifier,
    return two models and two Onehotencoder vectors
  """
    # Load the dataframe of train data
    df_train = read_in_preprocessed_data(datafile)

    # Model 1 Training:
    X_train = df_train[selected_features_1st]
    Y_train = df_train['1st_label']
    # Train the first model
    model_1, ohe_1 = argument_identifier(X_train, Y_train)

    # Model 2 Training: select a subset of rows that are arguments
    df_train_2nd = df_train[df_train['1st_label'] == 'Yes']
    X_train = df_train_2nd[selected_features_2nd]
    Y_train = list(df_train_2nd['2nd_label'])
    # Train the second model
    model_2, ohe_2 = argument_classifier(X_train, Y_train)

    return model_1, ohe_1, model_2, ohe_2


def evaluate(train_pre_path, test_pre_path, selected_features_1st=['lemma', 'word', 'pos1', 'pos2', 'predicate', 'predicate_pos1', 'predicate_pos2', 'basic_dependency',
    'extended_dependency', 'sentence_position', 'head_number'],
              selected_features_2nd=['lemma', 'word', 'pos1', 'pos2', 'predicate', 'predicate_pos1', 'predicate_pos2', 'basic_dependency',
    'extended_dependency', 'sentence_position', 'head_number'], preprocessed=False):
    """
    Trains models and gives classification report of both the first and second model.
  Input: 1. path of preprocessed training data; 2. path of preprocessed testing data
  3. two selected feature lists for each classifier
  """

    if not preprocessed:
        train_pre_path = preprocess(train_pre_path)
        test_pre_path = preprocess(test_pre_path)


    model_1, ohe_1, model_2, ohe_2 = training_pipeline(train_pre_path,
                                                       selected_features_1st,
                                                       selected_features_2nd)


    # First model evaluation:
    # Load the dataframe of test data
    df_test = read_in_preprocessed_data(test_pre_path)
    X_test = df_test[selected_features_1st]
    Y_test = df_test['1st_label']
    
    # Transform and predict
    X_test_ohe = ohe_1.transform(X_test)
    Y_pred = model_1.predict(X_test_ohe)
    df_test['1st prediction'] = np.array(Y_pred)
    report_1 = classification_report(Y_test, Y_pred, digits=3)
    print('First classification report: ')
    print(report_1)
    print()

    # Second model evaluation:
    # Get a subset of dataframe of rows predicted as argument (['1st prediction'] is Yes)
    df_test_2 = df_test[df_test['1st prediction'] == 'Yes']
    print(df_test_2)
    X_test = df_test_2[selected_features_2nd]
    print(X_test)
    Y_test = df_test_2['2nd_label']

    # Transform and predict
    X_test_ohe = ohe_2.transform(X_test)
    Y_pred = model_2.predict(X_test_ohe)
    df_test_2['2nd prediction'] = np.array(Y_pred)
    write_predictions_to_file(df_test_2, 'en_ewt-up-test.predictions.conllu')
    report_2 = classification_report(Y_test, Y_pred, digits = 3)
    print('Second classification report: ')
    print(report_2)


def main():
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    preprocess = sys.argv[3]
    
    if preprocess == 'no':
        evaluate(train_path, test_path)
    elif preprocess == 'yes':
        evaluate(train_path, test_path, preprocessed=True)


if __name__ == "__main__":
    main()

