import sys
import pandas as pd
# see tips & tricks on using defaultdict (remove when you do not use it)
from collections import defaultdict, Counter
# module for verifying output
#from nose.tools import assert_equal


def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file

    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string
    :returns: the annotations as a list
    '''
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    conll_input = pd.read_csv(inputfile, sep=delimiter, on_bad_lines='skip')
    annotations = conll_input[annotationcolumn].tolist()
    return annotations


def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output

    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations

    :returns: a countainer providing the counts for each predicted and gold class pair
    '''

    # TIP on how to get the counts for each class
    # https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)
    for gold, machineoutput in zip(goldannotations,machineannotations):
        evaluation_counts[gold][machineoutput] += 1
    return evaluation_counts

def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts

    :returns the precision, recall and f-score of each class in a container
    '''

    # TIP: you may want to write a separate function that provides an overview of true positives, false positives and false negatives
    #      for each class based on the outcome of obtain counts
    # YOUR CODE HERE (and remove statement below)
    #raise NotImplementedError()
    #Precision
    all_classes = evaluation_counts.keys()

    True_positive = []
    True_negative = []
    False_positive = []
    False_negative = []

    for cls in all_classes:
        fp = []
        tn = []
        for key, counter in evaluation_counts.items():
            if key == cls:
                tp = counter[cls]
                True_positive.append(tp)
                fn = 0
                for k, value in counter.items():
                    if k != cls:
                        fn += value
                False_negative.append(fn)
            else:
                for k, value in counter.items():
                    if k == cls:
                        fp.append(value)
                    else:
                        tn.append(value)
        False_positive.append(sum(fp))
        True_negative.append(sum(tn))

    result = {}
    for cls, tp, tn, fp, fn in zip(all_classes, True_positive, True_negative, False_positive, False_negative):
        rs_dict = defaultdict(dict)
        #? if tp == 0 is sufficient for the problem of zero division?
        # None means: this label doesn't appear in y_true
        if tp == 0:
            rs_dict['precision'] = 0
            rs_dict['recall'] = 0
            rs_dict['f1_socre'] = 0
            result[cls] = rs_dict
        else:
            precision = tp / (tp+fp)
            recall = tp / (tp+fn)
            rs_dict['precision'] = round(precision,3)
            rs_dict['recall'] = round(recall,3)
            rs_dict['f1_socre'] = round(2 * (precision * recall) / (precision + recall), 3)
            result[cls] = rs_dict
    return result

def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts

    :prints out a confusion matrix
    '''

    # TIP: provide_output_tables does something similar, but those tables are assuming one additional nested layer
    #      your solution can thus be a simpler version of the one provided in provide_output_tables below

    # YOUR CODE HERE (and remove statement below)
    all_classes = list(evaluation_counts.keys())
    confusion_matrix = [all_classes]
    # horizontal: system, vertical: gold
    # for index, cls in enumerate(all_classes):
    #     print(index, cls)
    #     y = list(0 for i in range(len(all_classes)))
    #     for key, counter in evaluation_counts.items():
    #         if cls in counter.keys():
    #             ind = all_classes.index(key)
    #             y[ind] = counter[cls]
    #     confusion_matrix.append(y)
    # return confusion_matrix

    # horizontal: gold, vertical: system
    # Use double for loop to make sure the iterate order
    for cls in all_classes:
        for key, counter in evaluation_counts.items():
            if key == cls:
                y = list(0 for i in range(len(all_classes)))
                for label, value in counter.items():
                    ind = all_classes.index(label)
                    y[ind] = value
                confusion_matrix.append(y)

    return confusion_matrix


def carry_out_evaluation(gold_annotations,
                         systemfile,
                         systemcolumn,
                         delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)

    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')

    returns evaluation information for this specific system
    '''
    system_annotations = extract_annotations(systemfile, systemcolumn,
                                             delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)

    return evaluation_outcome


def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems

    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict(
        {(i, j): evaluations[i][j]
         for i in evaluations.keys() for j in evaluations[i].keys()},
        orient='index')
    print(evaluations_pddf)
    print(evaluations_pddf.to_latex())


def run_evaluations(goldfile, goldcolumn, systems):
    '''
    Carry out standard evaluation for one or more system outputs

    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)

    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0],
                                              system[1])
        evaluations[system[2]] = sys_evaluation
    return evaluations
