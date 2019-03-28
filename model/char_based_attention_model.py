import torch
import torch.nn as nn
import json
from model.constants import *
class CharBasedAttentionRNN(nn.Module):
    def __init__(self, input_size=33, hidden_size=33, num_classes=3):
        super(CharBasedAttentionRNN, self).__init__()
        self.name = "CharBasedAttentionRNN"
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.softmax = torch.nn.Softmax()
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def attention(self, lstm_output, final_hidden_state):
        hidden = final_hidden_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output.to(self.device), hidden.unsqueeze(2).to(self.device)).squeeze(2)
        soft_attn_weights = self.softmax(attn_weights)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2).to(self.device), soft_attn_weights.unsqueeze(2).to(self.device)).squeeze(2)

        return new_hidden_state

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
            c0 = torch.zeros(1, 1, self.hidden_size)
        else:
            input = self.data_to_one_hot(x)
            h0 = torch.zeros(1, input.size(0), self.hidden_size)
            c0 = torch.zeros(1, input.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, (final_hidden_state, _) = self.rnn(input.to(self.device), (h0.to(self.device), c0.to(self.device)))
        out = self.attention(out, final_hidden_state)
        # Pass the output of the last time step to the classifier
        out = self.fc(out)
        return out

    def data_to_one_hot(self,x):
        ident = torch.eye(CHAR_HIDDEN)
        return ident[x]

    def one_hot_to_data(self,x):
        return torch.argmax(x)

    def canProcess(self, token):
        return True