import random, time, os
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import logging, sys
from transformers import BertTokenizer
import json
from typing import List, Dict, Tuple

import ym__bert_utils as utils


# LOAD DATASETS

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
        pred_position = []
        for row in sentence:
            if row[10] != "_" and row[10] != '':
                predicates.append(row[10])
                pred_position.append(row[0])
                #labels of the first predicate

        for ind, (predicate, position) in enumerate(zip(predicates, pred_position)):

            dup_sentences = []
            for row in sentence:

                if len(row) == len(predicates) + 11 and str(row[6]).isdigit():
                    split_row = row[:11]  #for each row in sentence, takes the first 11 columns
                    # append predicate information
                    split_row[10] = predicate
                    split_row.append(position)
                    split_row.append(row[10 + (ind + 1)])  #then appends the corresponding arguments column (note 'num in range(count)' starts at 0, hence the '+1' here)
                    dup_sentences.append(
                        split_row)  #append all rows of sentence to a list

            split_list.append(
                dup_sentences)  #append sentence to the full split_list

    return split_list

def create_json(filepath, outpath):

  split_text = split_predicates(filepath)

  list_of_sents = []
  for sent in split_text:
    sent_dict = {"seq_words":[], "BIO":[], "pred_sense":[]}
    for row in sent:
        sent_dict["seq_words"].append(row[1])
        sent_dict["BIO"].append(row[12])
    sent_dict["pred_sense"].append(sent[0][11])
    sent_dict["pred_sense"].append(sent[0][10])
    
    list_of_sents.append(sent_dict)

    with open(outpath, 'w') as outfile:
        json_object = json.dumps(list_of_sents)
        outfile.write(json_object)

def read_json_srl(filename: str) -> Tuple[List, List, Dict]:
    """ Read in json file as SRL data"""
    all_sentences, all_labels, all_pred_info = [], [], []
    label_dict = {}
    
    with open(filename, 'r') as openfile:
        data = json.load(openfile)
    
    for sentence in data:
        pred_info = []
        all_sentences.append(sentence['seq_words'])
        all_labels.append(sentence['BIO'])
    
        for label in sentence['BIO']:
            if label not in label_dict:
                label_dict[label] = len(label_dict)
        
        index, pred = sentence['pred_sense']
        index = int(index)
        for token in sentence['seq_words']:
            if len(pred_info) == index:
                pred_info.append(1)
            else:
                pred_info.append(0)
                
        all_pred_info.append(pred_info)
            
    return all_sentences, all_labels, label_dict, all_pred_info




def train(TRAIN_PATH, DEV_PATH, SAVE_MODEL_DIR):
    """ Train and save BERT model for SRL
    @param TRAIN_PATH: path to the training dataset
    @param DEV_PATH: path to the development dataset (used for validation)
    @param SAVE_MODEL_DIR: path to where the fine-tuned model will be saved """
    EPOCHS = 2
    BERT_MODEL_NAME = 'bert-base-multilingual-cased' #"bert-base-cased"
    GPU_RUN_IX=0

    SEED_VAL = 1234500
    SEQ_MAX_LEN = 256
    PRINT_INFO_EVERY = 10 # Print status only every X batches
    GRADIENT_CLIP = 1.0
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 4

    LABELS_FILENAME = f"{SAVE_MODEL_DIR}/label2index.json"
    LOSS_TRN_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Train_{EPOCHS}.json"
    LOSS_DEV_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Dev_{EPOCHS}.json"

    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)


    # Initialize Random seeds and validate if there's a GPU available...
    device, USE_CUDA = utils.get_torch_device(GPU_RUN_IX)
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # LOG FILE

    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{SAVE_MODEL_DIR}/BERT_TokenClassifier_train_{EPOCHS}.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logging.info("Start Logging")

    # TOKENIZER

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)

    # Load Train Dataset
    train_data, train_labels, train_label2index, train_preds_info = read_json_srl(TRAIN_PATH)
    train_inputs, train_masks, train_labels, train_seq_lengths, train_preds_info = utils.data_to_tensors(dataset=train_data, 
                                                                                tokenizer=tokenizer, 
                                                                                pred_info=train_preds_info, 
                                                                                max_len=SEQ_MAX_LEN, 
                                                                                labels=train_labels, 
                                                                                label2index=train_label2index,
                                                                                pad_token_label_id=PAD_TOKEN_LABEL_ID 
                                                                                )


    utils.save_label_dict(train_label2index, filename=LABELS_FILENAME)
    index2label = {v: k for k, v in train_label2index.items()}

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_preds_info)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Load Dev Dataset
    dev_data, dev_labels, dev_label2index , dev_preds_info = read_json_srl(DEV_PATH)
    dev_inputs, dev_masks, dev_labels, dev_seq_lengths, dev_preds_info = utils.data_to_tensors(dataset=dev_data, 
                                                                        tokenizer=tokenizer, 
                                                                        pred_info=dev_preds_info,
                                                                        max_len=SEQ_MAX_LEN, 
                                                                        labels=dev_labels, 
                                                                        label2index=dev_label2index,
                                                                        pad_token_label_id=PAD_TOKEN_LABEL_ID
                                                                        )

    # Create the DataLoader for our Development set.
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_preds_info)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

    # INITIALIZE MODEL

    model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(train_label2index))
    model.config.finetuning_task = 'token-classification'
    model.config.id2label = index2label
    model.config.label2id = train_label2index
    if USE_CUDA: model.cuda()

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS

    # Create optimizer and the learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

    # TRAIN

    loss_trn_values, loss_dev_values = [], []


    for epoch_i in range(1, EPOCHS+1):
        # Perform one full pass over the training set.
        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, EPOCHS))
        logging.info('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_predicates = batch[3].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, token_type_ids = b_predicates)
            loss = outputs[0]
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Progress update
            if step % PRINT_INFO_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = utils.format_time(time.time() - t0)
                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step, len(train_dataloader),
                                                                                                elapsed, loss.item()))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_trn_values.append(avg_train_loss)

        logging.info("")
        logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
        logging.info("  Training Epoch took: {:}".format(utils.format_time(time.time() - t0)))

        # VALIDATION
        # After the completion of each training epoch, measure our performance on our validation set.
        t0 = time.time()
        results, preds_list = utils.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, index2label, PAD_TOKEN_LABEL_ID, prefix="Validation Set")
        loss_dev_values.append(results['loss'])
        logging.info("  Validation Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))
        logging.info("  Validation took: {:}".format(utils.format_time(time.time() - t0)))


        # Save Checkpoint for this Epoch
        utils.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)


    utils.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
    utils.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)
    logging.info("")
    logging.info("Training complete!")





def main():
    TRAIN_PATH = sys.argv[1]
    DEV_PATH = sys.argv[2]
    SAVE_MODEL_DIR = sys.argv[3]

    train(TRAIN_PATH, DEV_PATH, SAVE_MODEL_DIR)


if __name__ == "__main__":
    main()