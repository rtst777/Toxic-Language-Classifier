import pandas as pd
import torchtext
import torch
import random
import numpy as np
import ast
import re
import json
import codecs
from model.baseline_model import ToxicBaseLSTM
from model.fasttext_based_lstm_model import FastTextBasedLSTMModel
from model.glove_based_lstm_model import GloveBasedLSTMModel
from model.glove_based_attention_lstm import GloveBasedAttentionLSTMModel
from model.fasttext_based_attention_lstm import FastTextBasedAttentionLSTMModel
from model.char_based_attention_model import CharBasedAttentionRNN
from model.char_based_model import Char_based_RNN
from model.constants import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model.train_model import set_global_seed, get_accuracy, get_model_name, tokenizer
import time
IS_CHAR = True

index_to_vocab = None
def create_dataloader():
    if not IS_CHAR:
        word_field = torchtext.data.Field(sequential=True,  # text sequence
                                          tokenize=tokenizer,  # because are building a character-RNN
                                          include_lengths=True,  # to track the length of sequences, for batc
                                          batch_first=True,
                                          use_vocab=True)  # to turn each character into an integer ind
    else:
        word_field = torchtext.data.Field(sequential=True,  # text sequence
                                          tokenize=lambda x: x,  # because are building a character-RNN
                                          include_lengths=True,  # to track the length of sequences, for batc
                                          batch_first=True,
                                          use_vocab=True)  # to turn each character into an integer ind

    label_field = torchtext.data.Field(sequential=False,  # not a sequence
                                       use_vocab=False,  # don't need to track vocabulary
                                       is_target=True,
                                       batch_first=True,
                                       preprocessing=lambda x: int(x))  # convert text to 0 and
    fields = [('data', word_field), ('label', label_field)]

    # dataset1 = torchtext.data.TabularDataset(path=github_cleaned_data, skip_header=True, format='csv', fields=fields)
    # TODO concatenate three datasets
    dataset2 = torchtext.data.TabularDataset(path=kaggle_cleaned_train_data, skip_header=True, format='csv', fields=fields)
    # dataset3 = torchtext.data.TabularDataset(path=kaggle_cleaned_test_data, skip_header=True, format='csv', fields=fields)
    #full_data = torchtext.data.TabularDataset(path=merged_cleaned_test_data, skip_header=True, format='csv',
     #                                        fields=fields)
    train_set, valid_set, test_set = split_data(dataset2)

    # create vocabulary index
    word_field.build_vocab(train_set)

    if IS_CHAR:
        global max_val
        max_val = max(word_field.vocab.stoi.values())
        with open('char_rnn_stoi.json', 'w') as f:
            json.dump(word_field.vocab.stoi,f)

    return train_set, valid_set, test_set


def split_data(dataset):
    train_set, valid_set, test_set = dataset.split([0.6, 0.2, 0.2], random_state=random.getstate())
    return train_set, valid_set, test_set



def evaluate(model, test_set, batch_size = 32):
    test_iter = torchtext.data.BucketIterator(test_set,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.data),  # to minimize padding
                                              sort_within_batch=True,  # sort within each batch
                                              repeat=False)  # repeat the iterator for

    correct, total = 0, 0
    total_time_cost = 0
    for i, batch in enumerate(test_iter):
        input_data = batch.data[0]
        start_time = time.time()
        output = model(input_data)  # Check this input data format
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time_cost += elapsed_time
        pred = output.max(1, keepdim=True)[1]
        labels = batch.label
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total, total_time_cost


if __name__== "__main__":
    set_global_seed()
    train_set, valid_set, test_set = create_dataloader()

    model = Char_based_RNN()
    saved_model_path = get_model_name(model.name, 256, 0.001, 18, 0.9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(saved_model_path, map_location=device))

    accuracy, total_time_cost = evaluate(model, test_set)
    print(("Model {}: Accuracy {}, total_time_cost {}").format(
        model.name,
        accuracy,
        total_time_cost))

