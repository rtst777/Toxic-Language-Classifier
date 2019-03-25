import torch
import torch.nn as nn
import json
from model.constants import *
class Char_based_RNN(nn.Module):
    def __init__(self, input_size=33, hidden_size=33, num_classes=3):
        super(Char_based_RNN, self).__init__()
        self.name = "Char_based_RNN"
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set an initial hidden state
        if (isinstance(x[0], str)):
            with open("char_rnn_stoi.json", 'r') as f:
                datastore = json.load(f)
            chars = []
            value = []
            for word in x:
                for c in word:
                    chars.append(c)
                    try:
                        value.append(datastore[c])
                    except:
                        value.append(0)
            input = torch.unsqueeze(torch.tensor(self.data_to_one_hot(value)),0)
            h0 = torch.zeros(1, 1, self.hidden_size)
        else:
            input = self.data_to_one_hot(x)
            h0 = torch.zeros(1, input.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(input, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(torch.max(out, dim=1)[0])
        return out

    def data_to_one_hot(self,x):
        ident = torch.eye(CHAR_HIDDEN)
        return ident[x]

    def one_hot_to_data(self,x):
        return torch.argmax(x)

    def canProcess(self, token):
        return True