# Project: 2023 code repo

## Advanced NLP
### 0228_SRL_two_clf_pipeline.py

__Task__: Semantic Role labeling through 1) argument identification LR classifier 2) argument classification classifier
__Packaged used__: sklearn logistic regression model and one hot encoder  <br> 
__Pipeline description__:
1) read in the data, extract features and labels;
2) store the features and labels in Pandas DataFrames and remove rows with predicates as tokens;
3) train the first argument identification classifier using all rows in the training DataFrame;
4) train the second argument classifier using a subset of rows that are arguments;
5) evaluate the first classifier on the evaluation dataset;
6) select a subset of evaluation data that were predicted as arguments by the first classifier and run evaluation on that subset.

 <br> 
 <br> 
 <br> 

### 0310 Neural SRL
__Task__: A BERT-based classifier for Semantic Role Labeling (SRL)  <br> 
__Packaged used__: Transformer BertForTokenClassfication  <br> 
__Pipeline description__: 
1) initializing the hyperparameters, setting elements such as number of epochs, learning rate, and batch size
2) Load the training and validation datasets, pad all sequences with attention mask tensor and padding predicate info

 <br> 
 <br> 
 <br> 

### 0430_Challenge_dataset_report.pdf
__Task__: Apply checklist evaluation methodology for behavioral testing of two AllenNLP semantic role labeling models (BERT-based and Bi-LSTM).
__Capabilities tested__:
1) identify high position arguments
2) Causative inchoative alteration
3) Subordinate clause
4) Long-span dependencies
5) Voice
6) NER
7) POS
8) Robustness
 <br> 
 <br> 

-------------
### 221223 Machine learning final submission
Experiment 0-8


 <br> 
 <br> 
 <br> 


## Reference
https://www.makeareadme.com/

## File description format <br> 
__Task__: <br> 
__Pipeline description__:  <br> 
__Packaged used__: <br> 
