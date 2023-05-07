Code to train and save a BERT model fine-tuned for SRL, and use that model to make and evaluate predictions


TRAINING

run:
python train.py [train_path] [dev_path] [model_path]

train_path = path to the training data
dev_path = path to the development/validation data
model_path = path to where the model should be saved

PREDICTING

run:
python predict.py [test_path] [model_path]

test_path = path to the test data
model_path = path to where the model should be saved