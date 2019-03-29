import torch
import torch.nn as nn

# model_and_score:
#   a list of tuple (model, score for that model)
#   e.g.
#       [(model1, 1), (models2, 2), (models3, 2)]
#
#   usage of the socre:
#       if the token can be handled by the model, the model get a score.
#       At the end, we apply "softmax" to the total scores of each model to get the weight for each model
#
# policy:
#  - average:   TODO
#       model 1 -> {class 1: 0.3, class 2: 0.3, class 3: 0.4}   weight: 0.3
#       model 2 -> {class 1: 0.2, class 2: 0.3, class 3: 0.5}   weight: 0.3
#       model 3 -> {class 1: 0.8, class 2: 0.1, class 3: 0.1}   weight: 0.4
#       prediction -> {class 1: 0.47, class 2: 0.22, class 3: 0.31}
#
#  - majority:  TODO
#       model 1 -> {class 1: 0.3, class 2: 0.3, class 3: 0.4}   weight: 0.3
#       model 2 -> {class 1: 0.2, class 2: 0.3, class 3: 0.5}   weight: 0.3
#       model 3 -> {class 1: 0.8, class 2: 0.1, class 3: 0.1}   weight: 0.4
#       prediction -> {class 1: 0.25, class 2: 0.3, class 3: 0.45}
#
#  - highest_weights:
#       model 1 -> {class 1: 0.3, class 2: 0.3, class 3: 0.4}   weight: 0.3
#       model 2 -> {class 1: 0.2, class 2: 0.3, class 3: 0.5}   weight: 0.3
#       model 3 -> {class 1: 0.8, class 2: 0.1, class 3: 0.1}   weight: 0.4
#       prediction -> {class 1: 0.8, class 2: 0.1, class 3: 0.1}
#
#  - mixture_of_experts:
#       most words have embedding -> word based model
#                                                     long sentence -> word based attention model
#                                                     else          -> word based non-attention model
#       else                      -> char based model
#                                                     long sentence -> char based attention model
#                                                     else          -> char based non-attention model
#
class EnsembleModels(nn.Module):
    def __init__(self, model_and_score, policy, word_based, char_based, word_attention_based, char_attention_based):
        super(EnsembleModels, self).__init__()
        self.name = 'EnsembleModels'
        self.model_and_score = model_and_score
        self.policy = policy
        self.word_based = word_based
        self.char_based = char_based
        self.word_attention_based = word_attention_based
        self.char_attention_based = char_attention_based

    def highest_weights(self, x, model_and_score, weights):
        idx = torch.argmax(weights)
        chosen_model = model_and_score[idx][0]
        return chosen_model(x)

    def mixture_of_experts(self, x):
        word_num_threshold = 20
        word_embedding_ratio = 0.9

        num_word_embedding = 0
        for token in x:
            if self.word_based.canProcess(token):
                num_word_embedding += 1

        chosen_model = None
        total_num_words = len(x)
        if num_word_embedding > total_num_words * word_embedding_ratio:
            if total_num_words > word_num_threshold:
                print("word_attention_based is used")
                chosen_model = self.word_attention_based
            else:
                print("word_attention is used")
                chosen_model = self.word_based
        else:
            if total_num_words > word_num_threshold:
                print("char_attention_based is used")
                chosen_model = self.char_attention_based
            else:
                print("char_based is used")
                chosen_model = self.char_based

        return chosen_model(x)

    def forward(self, x):
        if (self.policy == "mixture_of_experts"):
            out = self.mixture_of_experts(x)
            return out

        # compute weights
        weights = torch.zeros(len(self.model_and_score))
        sum = 0
        for i, item in enumerate(self.model_and_score):
            model = item[0]
            score = item[1]
            for token in x:
                if model.canProcess(token):
                    weights[i] += score
                    sum += score

        weights /= sum

        # make prediction based on policy
        if (self.policy == "highest_weights"):
            out = self.highest_weights(x, self.model_and_score, weights)
        elif (self.policy == "majority"):
            # TODO
            pass
        elif (self.policy == "average"):
            # TODO
            pass

        return out