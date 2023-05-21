# Project: 2023 code repo


### 0430 AI and ethics project: a group project probing into ChatGPT's biases
(Part 4: technical approach to ChatGPT's components made by Lisa)

 -----------------------


### 0430_Challenge_dataset_report.pdf
__Task__: Apply checklist evaluation methodology for behavioral testing of two AllenNLP semantic role labeling models (BERT-based and Bi-LSTM).
<br>
<br>
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

 -----------------------

### 0310 Neural SRL
__Task__: A BERT-based classifier for Semantic Role Labeling (SRL)  <br>
__Packaged used__: Transformer BertForTokenClassfication  <br>
__Pipeline description__:
1) initializing the hyperparameters, setting elements such as number of epochs, learning rate, and batch size
2) Load the training and validation datasets, pad all sequences with attention mask tensor and padding predicate info
<br>
 ------------------------------

### 0228_SRL_two_clf_pipeline.py
__Task__: Semantic Role labeling through 1) argument identification LR classifier 2) argument classification classifier <br>
__Packaged used__: sklearn logistic regression model and one hot encoder  <br>
__Pipeline description__:
1) read in the data, extract features and labels;
2) store the features and labels in Pandas DataFrames and remove rows with predicates as tokens;
3) train the first argument identification classifier using all rows in the training DataFrame;
4) train the second argument classifier using a subset of rows that are arguments;
5) evaluate the first classifier on the evaluation dataset;
6) select a subset of evaluation data that were predicted as arguments by the first classifier and run evaluation on that subset.
<br>

--------------------------------

### 1223 Machine learning final submission
__Task__: This task aims to investigate the performance of different feature combinations and machine learning algorithms by experimenting with models for the CoNLL-2003 named-entity recognition shared task. During the experiments, we added new informative features, and trained SVM, Naive Bayes, Logistic regression, BERT, LSTM and CRF models on the training data, and ran them on the development data.

*__Folder 1223_Experiments__*
#### 0. Feature engineering.ipynb
- contain the code that adds new features to the conll dataset (e.g. left, right token)
*Below notebooks were organized roughly based on the time sequence of each experiment.
<br>
The __development dataset__ was used in each notebook for evaluation.*
#### 1. A basic system.ipynb
#### 2. More features OHE & NB.ipynb
#### 3. Hypertuning.ipynb
#### 4. Word_Embeddings.ipynb
#### 5. Feature Ablaton.ipynb
#### 6. Dense & Sparse features.ipynb
#### 7. CRF_dev_evaluation.ipynb
#### 7. CRF.py
- original crf script from assignment3
#### 7. crf_dev_output.csv
- output from executing 'CRF.py' on dev data
#### 8. BERT_finetuner.ipynb

<br />

*__Folder 1223_Experiments.util__*
#### analysis_distribution.py
- this script is to plot labels of train/dev dataset distribution
#### a_mixed_system.py
- script from containing functions combining dense and sparse features
#### basic_evaluation.py
- script from assignment1
#### basic_system.py
- script from assignment 1
#### feature_extract.py
- this script contains functions to extract left/right token, and assign capitalization type to each token
#### ner_machine_learning.py

<br>

--------------------------------

## Reference
https://www.makeareadme.com/

## File description format <br>
__Task__: <br>
__Pipeline description__:  <br>
__Packaged used__: <br>
