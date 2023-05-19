#This is a script analyzing the label distribution in the data
#input argument: path to the respective data split
#output: a csv file representing the distribution of the labels

#Train: Train
#Evaluation: Dev


from basic_system import extract_features_and_labels
import matplotlib.pyplot as plt
import pandas as pd

train_path = '../../data/conll2003.train.conll'
dev_path = '../../data/conll2003.dev.conll'

X_train, Y_train = extract_features_and_labels(train_path)
X_test, Y_test = extract_features_and_labels(dev_path)

# Count label and output as a csv
Y_train = pd.Series(Y_train).value_counts()
Y_test = pd.Series(Y_test).value_counts()
df = pd.concat([Y_train, Y_test], axis = 1, names = ['Index','Train', 'Dev'])
df = df.rename(columns= {0:'Train', 1:'Dev'})
#df.to_csv('result.csv')


# check token distribution
#X_train = pd.Series(X_train).value_counts()

# Data visuliation
#df.plot.pie(subplots=True, figsize =(11,6))
df.plot.bar(subplots=True)

plt.show()
