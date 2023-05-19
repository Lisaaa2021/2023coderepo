from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
from nltk.stem import WordNetLemmatizer


def extract_features_and_labels(trainingfile):

    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token': token}
                feature_dict['PoS'] = components[1]
                feature_dict['Chunk_tag'] = components[2]
                data.append(feature_dict)
                targets.append(components[-1])
    return data, targets


def extract_features(inputfile):

    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token': token}
                data.append(feature_dict)
    return data



def create_classifier(train_features, train_targets):

    logreg = LogisticRegression(max_iter = 100000)
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    model = logreg.fit(features_vectorized, train_targets)

    return model, vec


def classify_data(model, vec, inputdata, outputfile):

    features = extract_features(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(
                line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()



def main(argv=None):

    #a very basic way for picking up commandline arguments
    if argv is None:
        argv = sys.argv

    #Note 1: argv[0] is the name of the python program if you run your program as: python program1.py arg1 arg2 arg3
    #Note 2: sys.argv is simple, but gets messy if you need it for anything else than basic scenarios with few arguments
    #you'll want to move to something better. e.g. argparse (easy to find online)

    #you can replace the values for these with paths to the appropriate files for now, e.g. by specifying values in argv
    #argv = ['mypython_program','','','']
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]

    training_features, gold_labels = extract_features_and_labels(trainingfile)
    ml_model, vec = create_classifier(training_features, gold_labels)
    classify_data(ml_model, vec, inputfile, outputfile)


#uncomment this when using this in a script

if __name__ == '__main__':
    main()

#print(sys.argv)
#when running this as 'main' in terminal, input:
# python basic_system.py  '../../data/conll2003.train.conll' '../../data/conll2003.dev.conll' 'result.csv'
