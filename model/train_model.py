import pandas as pd
import torchtext
import torch
import random
import numpy as np

github_data_clean_data = "../data/cleaned_data/dataset1.csv"


def set_global_seed(seed=37):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

glove = None
def convert_word_to_glove(text):
    global glove
    if glove is None:
        glove = torchtext.vocab.GloVe(name="6B", dim=50)  # TODO tune number

    glove_text = [glove[x] for x in text]
    return glove_text


def load_data_and_convert_to_glove():
    df = pd.read_csv(github_data_clean_data)
    subdf = df[['data', 'label']]
    subdf['data'] = subdf['data'].apply(lambda x: convert_word_to_glove(x))

    converted_data = subdf[['data']].values
    label_numpy = subdf[['label']].values
    converted_label = [torch.from_numpy(label).squeeze(0) for label in label_numpy]

    return list(zip(converted_data, converted_label))


# split data to train, valid, test set.
def get_splitted_data(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    data_size = len(dataset)

    # shuffle data set
    random.shuffle(dataset)

    # generate the index for randomly splitted training, validation, and test dataset
    train_val_boundary = int(data_size * train_ratio)
    val_test_boundary = int(data_size * (train_ratio + val_ratio))
    end_boundary = data_size
    train_data_set = dataset[0:train_val_boundary]
    val_data_set = dataset[train_val_boundary:val_test_boundary]
    test_data_set = dataset[val_test_boundary:end_boundary]

    return train_data_set, val_data_set, test_data_set


def get_data_loader(dataset):
    data, label = zip(*dataset)
    data = list(data)
    label = list(label)


    return torch.utils.data.TensorDataset(data, label)



if __name__== "__main__":
    set_global_seed()
    data_set = load_data_and_convert_to_glove()
    train_data_set, valid_data_set, test_data_set = get_splitted_data(data_set)
    train_data_loader = get_data_loader(train_data_set)
    valid_data_loader = get_data_loader(valid_data_set)
    test_data_loader = get_data_loader(test_data_set)

    print(len(train_data_loader))
    print(len(valid_data_loader))
    print(len(test_data_loader))


