from model.fasttext_based_lstm_model import FastTextBasedLSTMModel
from model.glove_based_lstm_model import GloveBasedLSTMModel
from model.glove_based_attention_lstm import GloveBasedAttentionLSTMModel
from model.glove_based_bidirection_lstm import GloveBasedBidirectionalLSTMModel
from model.fasttext_based_attention_lstm import FastTextBasedAttentionLSTMModel
from model.char_based_attention_model import CharBasedAttentionRNN
from model.char_based_model import Char_based_RNN
from model.constants import *
from model.train_model import set_global_seed, get_model_name, tokenizer
from server.service.classification_service import preprocess_input
import torch
from torch import nn
import json
import numpy

test_sentence = [
    (["bitch"], "offensive"),
    (["hi"], "neither"),
    (["hi dog"], "neither"),
    (["your have dick face"], "offensive"),
    (["your brain is in your ass"], "offensive"),
    (["sleep your mom"], "offensive"),
]

label = ["hate", "offensive", "neither"]
def predict(model, raw_input):
    clean_input = preprocess_input(raw_input)
    output = model(clean_input)
    softmax = nn.Softmax()
    output = softmax(output)
    predicted_prob, predicted_idx = torch.max(output, 1)
    predicted_label = label[predicted_idx.detach().numpy()[(0)]]
    return predicted_label

def run_smoke_test(model):
    print(("==================== Start to test Model \"{}\" ====================").format(model.name))
    all_pass = True
    for item in test_sentence:
        sentence = item[0]
        label = item[1]
        prediction = predict(model, sentence)
        if label != prediction:
            print(("failed on the sentence: \"{}\"").format(sentence[0]))
            print(("expected: \"{}\", but actually got \"{}\" \n").format(label, prediction))
            all_pass = False

    if all_pass:
        print("All tests passed")
    print(("======================================================================================\n").format(model.name))

if __name__== "__main__":
    # create model
    word_based_model = GloveBasedBidirectionalLSTMModel()
    saved_word_based_model_path = get_model_name(word_based_model.name, 32, 0.001, 11, 0.9)
    word_based_model.load_state_dict(torch.load(saved_word_based_model_path, map_location=device))

    # run tests
    run_smoke_test(word_based_model)