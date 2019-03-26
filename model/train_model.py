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
IS_CHAR = True

def set_global_seed(seed=37):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_accuracy(model, data, criterion, batch_size):
    data_iter = torchtext.data.BucketIterator(data,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.data),  # to minimize padding
                                              sort_within_batch=True,  # sort within each batch
                                              repeat=False)  # repeat the iterator for

    correct, total = 0, 0
    total_valid_loss = 0
    for i, batch in enumerate(data_iter):
        input_data = batch.data[0]
        output = model(input_data) # Check this input data format
        pred = output.max(1, keepdim=True)[1]
        labels = batch.label.to(device)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
        loss = criterion(output, labels)
        total_valid_loss += loss.item()
    return correct / total, float(total_valid_loss) /(i+1)

def get_model_name(name, batch_size, learning_rate, epoch,momentum):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}_momentum_{4}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch,momentum)
    return path

def train_model(model, train_set, valid_set, batch_size = 32, learning_rate = 0.001, num_epochs = 30, momentum = 0.9):
    train_iter = torchtext.data.BucketIterator(train_set,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.data),  # to minimize padding
                                              sort_within_batch=True,  # sort within each batch
                                              repeat=False)  # repeat the iterator for

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999))
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_iter):
            input_data = batch.data[0]
            optimizer.zero_grad()
            outputs = model(input_data)
            labels = batch.label.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_err[epoch], train_loss[epoch] = get_accuracy(model, train_set, criterion, batch_size)
        val_err[epoch], val_loss[epoch] = get_accuracy(model, valid_set, criterion, batch_size)
        print(("Epoch {}: Train accuracy: {}, Train loss: {} |"+
               "Validation accuracy: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(model.name, batch_size, learning_rate, epoch, momentum)
        torch.save(model.state_dict(), model_path)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)  # FIXME  err should be accuracy
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
    print('Finished Training')


def tokenizer(word_str):
    list_of_word = ast.literal_eval(codecs.encode(word_str).decode("utf-8"))
    return list_of_word
    # return " ".join(list_of_word)
    # return word_str
    # return convert_word_to_glove(list_of_word)

def balance_data_set(data_set):
    # save the original training examples
    old_examples = data_set.examples

    train_non_toxic_example = []
    train_offensive_toxic_example = []
    train_hate_speech_toxic_example = []
    for item in data_set.examples:
        if item.label == 2:
            train_non_toxic_example.append(item)
        elif item.label == 0:
            train_hate_speech_toxic_example.append(item)
        elif item.label == 1:
            train_offensive_toxic_example.append(item)


    total_num_non_toxic_example = len(train_non_toxic_example)
    total_num_offensive_example = len(train_offensive_toxic_example)
    total_num_hate_speech_toxic_example = len(train_hate_speech_toxic_example)

    # duplicate non-toxic and hate speech example
    non_toxic_scale_factor = (total_num_offensive_example // total_num_non_toxic_example) - 1
    hate_speech_toxic_scale_factor = (total_num_offensive_example // total_num_hate_speech_toxic_example) - 1
    data_set.examples = old_examples + train_non_toxic_example * non_toxic_scale_factor + train_hate_speech_toxic_example * hate_speech_toxic_scale_factor

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

    #dataset1 = torchtext.data.TabularDataset(path=github_cleaned_data, skip_header=True, format='csv', fields=fields)
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

    global index_to_vocab
    index_to_vocab = word_field.vocab.itos
    # balance training set data
    balance_data_set(train_set)

    return train_set, valid_set, test_set


def split_data(dataset):
    train_set, valid_set, test_set = dataset.split([0.6, 0.2, 0.2], random_state=random.getstate())
    return train_set, valid_set, test_set

def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Accuracy")
    plt.plot(range(1,31), train_err, label="Train")
    plt.plot(range(1,31), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,31), train_loss, label="Train")
    plt.plot(range(1,31), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


if __name__== "__main__":
    set_global_seed()
    train_set, valid_set, test_set = create_dataloader()

    #model = GloveBasedAttentionLSTMModel(index_to_vocab=index_to_vocab)
    #train_model(model, train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9) # Epoch 30: Train accuracy: 0.8516316874924178, Train loss: 0.39275846587698166 |Validation accuracy: 0.8214285714285714, Validation loss: 0.4745837217377078

    # model = GloveBasedLSTMModel(index_to_vocab=index_to_vocab)
    # train_model(model, train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9) # 0.866711

    # model = FastTextBasedLSTMModel(index_to_vocab=index_to_vocab)
    # train_model(model, train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9) # Epoch 30: Train accuracy: 0.9196421407365802, Train loss: 0.264950641202567 |Validation accuracy: 0.8660209846650525, Validation loss: 0.3871996518104307
    print(max_val+1)
    #model = Char_based_RNN(max_val+1,max_val+1,3)
    #train_model(model.cuda(), train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9)
    # train_model(model, train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9) # Epoch 17: Train accuracy: 0.7729588741962877, Train loss: 0.5484655738801891 |Validation accuracy: 0.8264729620661824, Validation loss: 0.45265432788479715
    #print(max_val+1)
    #model = Char_based_RNN(max_val+1,max_val+1,3)
    #train_model(model, train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9)
    # model = Char_based_RNN(33, 33, 3)
    # model.load_state_dict(torch.load('model_Char_based_RNN_bs32_lr0.001_epoch29_momentum_0.9'))
    # output = model(['hello'])
    # print(output)
    # output = model(['bitches', 'get', 'cut', 'everyday', 'b'])
    # print(output)
    #train_model(model.cuda(), train_set, valid_set, batch_size=256, learning_rate=0.001, num_epochs=30, momentum=0.9)
    model = CharBasedAttentionRNN()
    train_model(model.cuda(), train_set, valid_set, batch_size=128, learning_rate=0.001, num_epochs=30, momentum=0.9)


    #model = Char_based_RNN(33, 33, 3).cuda()
    #model.load_state_dict(torch.load('model_Char_based_RNN_bs256_lr0.001_epoch18_momentum_0.9'))
    #model = Char_based_RNN(33, 33, 3).to(device)
    #model.load_state_dict(torch.load('model_Char_based_RNN_bs256_lr0.001_epoch18_momentum_0.9'))
    #criterion = nn.CrossEntropyLoss()
    #err,loss = get_accuracy(model, test_set , criterion, 32)
    #print(err)
    #output = model(['hello'])
    #print(output)
    #output = model(['bitches', 'get', 'shit', 'everyday', 'b'])
    #print(output)
    #plot_training_curve(get_model_name("Char_based_RNN", 32, 0.001, 29, 0.9))

