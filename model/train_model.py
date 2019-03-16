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
from model.fasttext_based_lstm_model import FastText_Based_LSTM_Model
from model.glove_based_lstm_model import Glove_Based_LSTM_Model
from model.constants import github_data_clean_data
import torch.nn as nn
import torch.optim as optim


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
        output = model(batch.data[0]) # Check this input data format
        pred = output.max(1, keepdim=True)[1]
        labels = batch.label
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
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_iter):
            input_data = batch.data[0]
            optimizer.zero_grad()
            outputs = model(input_data)
            labels = batch.label
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
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
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

index_to_vocab = None
def create_dataloader():
    word_field = torchtext.data.Field(sequential=True,  # text sequence
                                      tokenize=tokenizer,  # because are building a character-RNN
                                      include_lengths=True,  # to track the length of sequences, for batc
                                      batch_first=True,
                                      use_vocab=True)  # to turn each character into an integer ind
    label_field = torchtext.data.Field(sequential=False,  # not a sequence
                                       use_vocab=False,  # don't need to track vocabulary
                                       is_target=True,
                                       batch_first=True,
                                       preprocessing=lambda x: int(x))  # convert text to 0 and
    fields = [('data', word_field), ('label', label_field)]

    dataset = torchtext.data.TabularDataset(path=github_data_clean_data, skip_header=True, format='csv', fields=fields)
    train_set, valid_set, test_set = split_data(dataset)
    word_field.build_vocab(train_set)
    global index_to_vocab
    index_to_vocab = word_field.vocab.itos

    return train_set, valid_set, test_set


def split_data(dataset):
    train_set, valid_set, test_set = dataset.split([0.6, 0.2, 0.2], random_state=random.getstate())
    return train_set, valid_set, test_set


if __name__== "__main__":
    set_global_seed()
    train_set, valid_set, test_set = create_dataloader()

    # model = Glove_Based_LSTM_Model(index_to_vocab=index_to_vocab)
    # train_model(model, train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9) # 0.866711

    model = FastText_Based_LSTM_Model(index_to_vocab=index_to_vocab)
    train_model(model, train_set, valid_set, batch_size=32, learning_rate=0.001, num_epochs=30, momentum=0.9) # Train accuracy: 0.8680564895763282, Train loss: 0.3627691795108139 |Validation accuracy: 0.8615819209039548, Validation loss: 0.3758530813840128


    # baseline_model = ToxicBaseLSTM()
    # saved_model_path = get_model_name(baseline_model.name, 32, 0.001, 0, 0.9)
    # baseline_model.load_state_dict(torch.load(saved_model_path))
    #

