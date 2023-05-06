# 2023coderepo
Lisa's code 2023

**0228_SRL_two_clf_pipeline.py**

Pipeline: We built a training and evaluation pipeline taking the training and evaluation datafiles in CoNLL-U Plus format and output two classification reports for each classifier following the below steps: 
1) read in the data, extract features and labels; 
2) store the features and labels in Pandas DataFrames and remove rows with predicates as tokens;
3) train the first argument identification classifier using all rows in the training DataFrame; 
4) train the second argument classifier using a subset of rows that are arguments; 
5) evaluate the first classifier on the evaluation dataset;
6) select a subset of evaluation data that were predicted as arguments by the first classifier and run evaluation on that subset.
