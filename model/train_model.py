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

    glove_text = [glove[x].tolist() for x in text]
    return glove_text


def load_data_and_convert_to_glove():
    df = pd.read_csv(github_data_clean_data)
    subdf = df[['data', 'label']]
    subdf['data'] = subdf['data'].apply(lambda x: convert_word_to_glove(x))

    converted_data = subdf[['data']].values
    converted_data = [arr.tolist() for arr in converted_data]
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

    data_tensor = torch.tensor(data)
    label_tensor = torch.tensor(label)

    return torch.utils.data.TensorDataset(data_tensor, label_tensor)

def get_accuracy(model, data, criterion, batch_size):
    data_iter = torchtext.data.BucketIterator(data,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.sms),
                                              repeat=False)
    correct, total = 0, 0
    total_valid_loss = 0
    for i, batch in enumerate(data_iter):
        output = model(batch.sms[0]) # Check this input data format
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(batch.label.view_as(pred)).sum().item()
        labels = batch.label
        total += batch.sms[1].shape[0]
        loss = criterion(output, labels.long())
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

def train_model(model, train, valid, batch_size = 32, learning_rate = 0.001, num_epochs = 30, momentum = 0.9):
    train_iterator = torchtext.data.BucketIterator(train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: len(x.sms), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                   # repeat the iterator for multiple epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_iterator):
            input_data = batch.sms[0]
            optimizer.zero_grad()
            outputs = model(input_data)
            labels = batch.label
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
        train_err[epoch], train_loss[epoch] = get_accuracy(model, train, criterion, batch_size)
        val_err[epoch], val_loss[epoch] = get_accuracy(model, valid, criterion, batch_size)
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


