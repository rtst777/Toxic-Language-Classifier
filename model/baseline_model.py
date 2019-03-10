import torch
import torch.nn as nn
from train_model import INPUT_SIZE
import torch.nn.functional as F

class ToxicBaseLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=INPUT_SIZE, num_classes=3):
        # one-hot embedding (preprocessing is outside of the model)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x is already in the format of glove vector
        # Need to call the function "convert_word_to_glove"
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the LSTM
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

