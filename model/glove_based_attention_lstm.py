import torch
import torch.nn as nn
from model.constants import GLOVE_INPUT_SIZE
import torchtext

class GloveBasedAttentionLSTMModel(nn.Module):
    def __init__(self, index_to_vocab=None, input_size=GLOVE_INPUT_SIZE, hidden_size=GLOVE_INPUT_SIZE, num_classes=3):
        super(GloveBasedAttentionLSTMModel, self).__init__()
        self.index_to_vocab = index_to_vocab
        self.glove = torchtext.vocab.GloVe(name="6B", dim=GLOVE_INPUT_SIZE)  # TODO tune number
        self.hidden_size = hidden_size
        self.name = 'GloveBasedAttentionLSTMModel'
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.softmax = torch.nn.Softmax()
        self.fc = nn.Linear(hidden_size, num_classes)


    def attention(self, lstm_output, final_hidden_state):
        hidden = final_hidden_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = self.softmax(attn_weights)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state


    def convert_input_to_glove(self, x):
        if isinstance(x, list):
            out = torch.zeros(1, len(x), GLOVE_INPUT_SIZE)
            for i in range(len(x)):
                out[0][i] = self.glove[x[i]]
        else:
            out = torch.zeros(x.shape[0], x.shape[1], GLOVE_INPUT_SIZE)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    out[i][j] = self.glove[self.index_to_vocab[x[i][j]]]

        return out

    def forward(self, x):
        x = self.convert_input_to_glove(x)
        # Need to call the function "convert_word_to_glove"
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the LSTM
        out, (final_hidden_state, _) = self.rnn(x, (h0, c0))
        attention_out = self.attention(out, final_hidden_state)
        out = self.fc(attention_out)
        return out

    def canProcess(self, token):
        return not torch.equal(self.glove[token], torch.zeros(GLOVE_INPUT_SIZE))
