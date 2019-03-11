from model.baseline_model import ToxicBaseLSTM
from model.dummy_model import DummyNet
from model.train_model import get_model_name, convert_word_to_glove
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
    baseline_model = ToxicBaseLSTM()
    saved_model_path = get_model_name(baseline_model.name, 32, 0.001, 0, 0.9)
    baseline_model.load_state_dict(torch.load(saved_model_path))
    return baseline_model
    # return DummyNet()


def preprocess_input(rawinput):
    rawinput = remove_punct(rawinput)
    rawinput = tokenization(rawinput)
    rawinput = remove_stopwords(rawinput)
    rawinput = remove_empty_string_token(rawinput)
    rawinput = lower_case_text(rawinput)
    clean_input = convert_word_to_glove(rawinput)
    # TODO might need to convert to GloVe embedding
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

    clean_input = torch.tensor(clean_input).unsqueeze(0)
    output = model(clean_input)
    softmax = nn.Softmax()
    output = softmax(output)
    predicted_prob, predicted_idx = torch.max(output, 1)
    predicted_label = label[predicted_idx.detach().numpy()[(0)]]
    confidence = predicted_prob.detach().numpy()[(0)]
    return json.dumps({"predicted_label": predicted_label, "confidence": confidence}, cls=MyEncoder)
