# Project: 2023 code repo

## Advanced NLP
### 0228_SRL_two_clf_pipeline.py

__Task__: Semantic Role labeling by two step classification

__Pipeline description__:
1) read in the data, extract features and labels;
2) store the features and labels in Pandas DataFrames and remove rows with predicates as tokens;
3) train the first argument identification classifier using all rows in the training DataFrame;
4) train the second argument classifier using a subset of rows that are arguments;
5) evaluate the first classifier on the evaluation dataset;
6) select a subset of evaluation data that were predicted as arguments by the first classifier and run evaluation on that subset.
__Packaged used__: sklearn logistic regression model and one hot encoder

### 0310 Neural SRL

### 0430_Challenge_dataset_report.pdf


-------------
### 221223 Machine learning final submission
Experiment 0-8


## Reference
https://www.makeareadme.com/

## File description format
__Task__:
__Pipeline description__: 
__Packaged used__:
