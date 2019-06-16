# Toxic-Language-Classifier
## Description
The project is an ensemble of models that classifies the input text messages as hate speech, offensive language or neither. The models are trained on the datasets collected from [comments in Wikipedia](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and [tweets](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data). The project could be applied on the public platform, for the purpose of automatic classification of toxic comments, and provide one fruitful and friendly discussion environment.

## Model Types
**Word Based LSTM**: 
* Bidirectional LSTM with glove embedding. 
* Provides distributed word representation. 
<img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image8.png" width="440" height="330">

**Character Based LSTM**:  
* Bidirectional LSTM with character encoding. 
* Handles typo or creative words.
<img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image2.png" width="400" height="290">

**Attention Based Word/Char Models**: 
* Attention based LSTM with glov embedding or character encoding. 
* Tracks long term dependency.
<img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image7.png" width="400" height="260">

**Ensemble**: We designed a customized ensemble strategy when building the overall software structure. First, check if most of words have glove embedding. If so, we use word-based model. Otherwise, we use character-based model. Then we check if it is a long sentence. If so, we replace the model with the attention-based model. Otherwise, we don’t use attention to save the prediction time.

<img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image14.png" width="630" height="400">


## User Interface
The classifier is served from the web server with friendly user interface.

<img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image4.png" width="220" height="350"> <img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image16.png" width="220" height="350"> <img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image17.png" width="220" height="350">
<img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image9.png" width="330" height="200">
<img src = "https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/images/image11.png" width="330" height="200">

## Prerequisite
`
Python(>=3.6.0)  
Pytorch(>=1.0.0)
`
## Installing


## Authors
Yizhan Jiang (yizhan.jiang@mail.utoronto.ca) <br>
Bo Li (clare.li@mail.utoronto.ca) <br>
Yunqi Huang (yunqi.huang@mail.utoronto.ca)


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/rtst777/Toxic-Language-Classifier/blob/develop/LICENSE) file for details


