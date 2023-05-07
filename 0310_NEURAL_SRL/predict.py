from transformers import pipeline
from torch.utils.data import SequentialSampler
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, AdamW
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import logging, sys

import ym__bert_utils as utils
from train import read_json_srl





def evaluate(TEST_PATH, MODEL_DIR):
    """
    Predicts SRL labels on test dataset using loaded fine-tuned model
    @param TEST_PATH: path to the test dataset
    @param MODEL_DIR: path to the saved fine-tuned model"""

    # Initializing parameters
    GPU_IX=0
    _, USE_CUDA = utils.get_torch_device(GPU_IX)
    FILE_HAS_GOLD = True
    SEQ_MAX_LEN = 256
    BATCH_SIZE = 4
    LOAD_EPOCH = 1
    INPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

    # Logging file
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/BERT_TokenClassifier_predictions.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

    # Load model and tokenizer
    model, tokenizer = utils.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
    label2index = utils.load_label_dict(f"{MODEL_DIR}/label2index.json")
    index2label = {v:k for k,v in label2index.items()}

    # Load Test Dataset
    prediction_data, prediction_labels, prediction_label2index, prediction_preds_info = read_json_srl(TEST_PATH)
    prediction_inputs, prediction_masks, gold_labels, _, prediction_preds_info = utils.data_to_tensors(dataset=prediction_data, 
                                                                                tokenizer=tokenizer, 
                                                                                pred_info=prediction_preds_info, 
                                                                                max_len=SEQ_MAX_LEN, 
                                                                                labels=prediction_labels, 
                                                                                label2index=prediction_label2index,
                                                                                pad_token_label_id=PAD_TOKEN_LABEL_ID 
                                                                                )


    # Evaluate 
    if FILE_HAS_GOLD:
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, prediction_preds_info)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

        logging.info('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
        
        results, preds_list = utils.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer, index2label, 
                                                        PAD_TOKEN_LABEL_ID, full_report=True, prefix="Test Set")

        logging.info("  Test Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))

        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for sent, pred in preds_list:
                    fin.write(" ".join(sent)+"\n")
                    fout.write(" ".join(pred)+"\n")

    else:
        # https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TokenClassificationPipeline
        logging.info('Predicting labels for {:,} test sentences...'.format(len(TEST_PATH)))
        if not USE_CUDA: GPU_IX = -1
        nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=GPU_IX)
        nlp.ignore_labels = []
        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for seq_ix, seq in enumerate(TEST_PATH):
                    sentence = " ".join(seq)
                    predicted_labels = []
                    output_obj = nlp(sentence)
                    # [print(o) for o in output_obj]
                    for tok in output_obj:
                        if '##' not in tok['word']:
                            predicted_labels.append(tok['entity'])
                    logging.info(f"\n----- {seq_ix+1} -----\n{seq}\nPRED:{predicted_labels}")
                    fin.write(sentence+"\n")
                    fout.write(" ".join(predicted_labels)+"\n")

def main():
    TEST_PATH = sys.argv[1]
    MODEL_DIR = sys.argv[2]

    evaluate(TEST_PATH, MODEL_DIR)


if __name__ == "__main__":
    main()
