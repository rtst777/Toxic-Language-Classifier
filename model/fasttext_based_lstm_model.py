import torch
import torch.nn as nn
from model.constants import FAST_TEXT_INPUT_SIZE
import torchtext

class FastTextBasedLSTMModel(nn.Module):
    def __init__(self, index_to_vocab=None, input_size=FAST_TEXT_INPUT_SIZE, hidden_size=FAST_TEXT_INPUT_SIZE, num_classes=3):
        super(FastTextBasedLSTMModel, self).__init__()
        self.index_to_vocab = index_to_vocab
        self.fasttext = torchtext.vocab.FastText(language='simple')  # TODO tune number
        self.hidden_size = hidden_size
        self.name = 'FastTextBasedLstmModel'
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def convert_input_to_fasttext(self, x):
        if isinstance(x, list):
            out = torch.zeros(1, len(x), FAST_TEXT_INPUT_SIZE)
            for i in range(len(x)):
                out[0][i] = self.fasttext[x[i]]
        else:
            out = torch.zeros(x.shape[0], x.shape[1], FAST_TEXT_INPUT_SIZE)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    out[i][j] = self.fasttext[self.index_to_vocab[x[i][j]]]

        return out

    def forward(self, x):
        x = self.convert_input_to_fasttext(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the LSTM
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def canProcess(self, token):
        return not torch.equal(self.fasttext[token], torch.zeros(FAST_TEXT_INPUT_SIZE))
