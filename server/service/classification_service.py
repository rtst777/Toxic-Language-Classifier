from model.dummy_model import DummyNet
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
    return DummyNet()


def preprocess_input(rawinput):
    rawinput = remove_punct(rawinput)
    rawinput = tokenization(rawinput)
    rawinput = remove_stopwords(rawinput)
    rawinput = remove_empty_string_token(rawinput)
    clean_input = lower_case_text(rawinput)
    # TODO might need to convert to GloVe embedding
    return clean_input

label = ["hate", "offensive", "neither"]
model = None
def predict(raw_input):
    global model
    if model is None:
        model = getModel()

    clean_input = preprocess_input(raw_input)
    output = model(clean_input)
    softmax = nn.Softmax()
    output = softmax(output)
    predicted_prob, predicted_idx = torch.max(output, 0)
    predicted_label = label[predicted_idx.numpy()[()]]
    confidence = predicted_prob.numpy()[()]
    return json.dumps({"predicted_label": predicted_label, "confidence": confidence}, cls=MyEncoder)
