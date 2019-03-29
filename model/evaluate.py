import pandas as pd
import torchtext
import torch
import random
import numpy as np
import ast
import re
import json
import codecs
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
from model.train_model import set_global_seed, get_model_name, tokenizer
import time
IS_CHAR = False
index_to_vocab = None

def create_test_set():
    word_num_threshold = 30
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
        word_num_threshold *= 10

    label_field = torchtext.data.Field(sequential=False,  # not a sequence
                                       use_vocab=False,  # don't need to track vocabulary
                                       is_target=True,
                                       batch_first=True,
                                       preprocessing=lambda x: int(x))  # convert text to 0 and
    fields = [('data', word_field), ('label', label_field)]

    test_dataset = torchtext.data.TabularDataset(path=kaggle_cleaned_test_data, skip_header=True, format='csv', fields=fields)
    word_field.build_vocab(test_dataset)
    global index_to_vocab
    index_to_vocab = word_field.vocab.itos

    test_set, _ = test_dataset.split([0.01, 0.99], random_state=random.getstate())
    test_set, long_test_set, short_test_set = test_set.split([0.4, 0.3, 0.3], random_state=random.getstate())

    prepare_long_sentence_test_set(long_test_set, word_num_threshold)
    prepare_short_sentence_test_set(short_test_set, word_num_threshold)

    return test_set, long_test_set, short_test_set


def prepare_long_sentence_test_set(data_set, word_num_threshold):
    long_sentence_example = []
    for item in data_set.examples:
        if len(item.data) > word_num_threshold:
            long_sentence_example.append(item)
    data_set.examples = long_sentence_example


def prepare_short_sentence_test_set(data_set, word_num_threshold):
    short_sentence_example = []
    for item in data_set.examples:
        if len(item.data) <= word_num_threshold:
            short_sentence_example.append(item)
    data_set.examples = short_sentence_example



def evaluate(model, test_set, batch_size = 1):
    test_iter = torchtext.data.BucketIterator(test_set,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.data),  # to minimize padding
                                              sort_within_batch=True,  # sort within each batch
                                              repeat=False)  # repeat the iterator for

    correct, total = 0, 0
    total_time_cost = 0
    for i, batch in enumerate(test_iter):
        input_data = batch.data[0]
        if input_data.shape[1] == 0:
            continue

        start_time = time.time()
        output = model(input_data)  # Check this input data format
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time_cost += elapsed_time
        pred = output.max(1, keepdim=True)[1]
        labels = batch.label
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]

    return correct / total, total_time_cost / total


def measure_model_performance(model, test_set, long_sentence_test_set, short_test_set):
    print(("==================== Performance Measurement on Model \"{}\" ====================").format(model.name))
    accuracy, time_cost_per_sentence = evaluate(model, test_set)
    print("Normal Test Set:")
    print(("Accuracy {}, Time Cost per Sentence {} seconds \n").format(accuracy, time_cost_per_sentence))

    accuracy, time_cost_per_sentence = evaluate(model, long_sentence_test_set)
    print("Long Sentence Test Set:")
    print(("Accuracy {}, Time Cost per Sentence {} seconds \n").format(accuracy, time_cost_per_sentence))

    accuracy, time_cost_per_sentence = evaluate(model, short_test_set)
    print("Short Sentence Test Set:")
    print(("Accuracy {}, Time Cost per Sentence {} seconds \n").format(accuracy, time_cost_per_sentence))

    print(("======================================================================================\n").format(model.name))



if __name__== "__main__":
    set_global_seed()
    test_set, long_sentence_test_set, short_test_set = create_test_set()

    # create models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # char_model = Char_based_RNN()
    # saved_char_model_path = get_model_name(char_model.name, 256, 0.001, 18, 0.9)
    # char_model.load_state_dict(torch.load(saved_char_model_path, map_location=device))
    #
    # char_attention_model = CharBasedAttentionRNN()
    # saved_char_attention_model_path = get_model_name(char_attention_model.name, 256, 0.001, 4, 0.9)
    # char_attention_model.load_state_dict(torch.load(saved_char_attention_model_path, map_location=device))

    word_based_model = GloveBasedLSTMModel(index_to_vocab=index_to_vocab)
    saved_word_based_model_path = get_model_name(word_based_model.name, 32, 0.001, 11, 0.9)
    word_based_model.load_state_dict(torch.load(saved_word_based_model_path, map_location=device))



    # measure model performance
    # measure_model_performance(char_model, test_set, long_sentence_test_set, short_test_set)
    # measure_model_performance(char_attention_model, test_set, long_sentence_test_set, short_test_set)
    measure_model_performance(word_based_model, test_set, long_sentence_test_set, short_test_set)