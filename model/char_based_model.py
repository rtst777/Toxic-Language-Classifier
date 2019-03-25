import torch
import torch.nn as nn

class CharBasedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=3):
        super(CharBasedRNN, self).__init__()
        self.name = "CharBasedRNN"
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set an initial hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(torch.max(out, dim=1)[0])
        return out

    def canProcess(self, token):
        return True
