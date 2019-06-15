# Toxic-Language-Classifier
## Description
The project is one ensemble of models (RNNs) that classifies the input text messages as hate speech, offensive language or neither. The project could be applied on the public platform, for the purpose of automatic classification of toxic comments, and provide one fruitful and friendly discussion environment.

**Word Based LSTM**: contains bidirectional LSTM layers followed by a fully connected layer, accepts the glove embedding of words as inputs, where the glove embedding provides distributed word representation. 

**Character Based LSTM**: built to address the issues raised in the words based model. This model used bidirectional LSTM followed by a fully connected layer as well and accepts the one-hot character encoding as inputs. This model was able to process typo or creative words

**Attention Based Word/Char Models**: after the last LSTM state is computed, going back and loop through all the previous LSTM states when making the prediction, which is so-called attention.

**Ensemble**: We designed a customized ensemble strategy when building the overall software structure. First, check if most of words have glove embedding. If so, we use word-based model. Otherwise, we use character-based model. Then we check if it is a long sentence. If so, we replace the model with the attention-based model. Otherwise, we don’t use attention to save the prediction time.

## Prerequisite
`
Python(>=3.6.0)  
Pytorch(>=1.0.0)
`
