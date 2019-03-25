from model.baseline_model import ToxicBaseLSTM
from model.dummy_model import DummyNet
from model.train_model import get_model_name
from model.glove_based_lstm_model import GloveBasedLSTMModel
from model.fasttext_based_lstm_model import FastTextBasedLSTMModel
from model.ensemble_models import EnsembleModels
from model.char_based_model import Char_based_RNN
from model.glove_based_attention_lstm import GloveBasedAttentionLSTMModel
from data.data_converter import remove_punct, tokenization, remove_stopwords, remove_empty_string_token, lower_case_text
import torch
from torch import nn
import json
import numpy

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def getModel():
    # model = GloveBasedLSTMModel()
    # model = FastTextBasedLSTMModel()
    # model = Char_based_RNN()
    model = GloveBasedAttentionLSTMModel()
    saved_model_path = get_model_name(model.name, 32, 0.001, 29, 0.9)
    model.load_state_dict(torch.load(saved_model_path))
    return model
#     # return DummyNet()


# def getModel():
#     model1 = GloveBasedLSTMModel()
#     saved_model_path1 = get_model_name(model1.name, 32, 0.001, 29, 0.9)
#     model1.load_state_dict(torch.load(saved_model_path1))
#
#     model2 = FastTextBasedLSTMModel()
#     saved_model_path2 = get_model_name(model2.name, 32, 0.001, 29, 0.9)
#     model2.load_state_dict(torch.load(saved_model_path2))
#
#     model3 = Char_based_RNN()
#     saved_model_path3 = get_model_name(model3.name, 32, 0.001, 29, 0.9)
#     model3.load_state_dict(torch.load(saved_model_path3))
#
#     model_and_score = [(model1, 2), (model2, 2), (model3, 1)]
#     policy = "highest_weights"
#
#     return EnsembleModels(model_and_score, policy)


def preprocess_input(rawinput):
    rawinput = remove_punct(rawinput)
    rawinput = tokenization(rawinput)
    rawinput = remove_stopwords(rawinput)
    rawinput = remove_empty_string_token(rawinput)
    clean_input = lower_case_text(rawinput)
    return clean_input

label = ["hate", "offensive", "neither"]
model = None
def predict(raw_input):
    global model
    if model is None:
        model = getModel()

    clean_input = preprocess_input(raw_input)
    if len(clean_input) == 0:
        return json.dumps({"predicted_label": "neither", "confidence": 0.900}, cls=MyEncoder)

    output = model(clean_input)
    softmax = nn.Softmax()
    output = softmax(output)
    predicted_prob, predicted_idx = torch.max(output, 1)
    predicted_label = label[predicted_idx.detach().numpy()[(0)]]
    confidence = predicted_prob.detach().numpy()[(0)]
    return json.dumps({"predicted_label": predicted_label, "confidence": confidence}, cls=MyEncoder)
