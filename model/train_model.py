import pandas as pd
import torchtext
import torch

github_data_clean_data = "../data/cleaned_data/dataset1.csv"

glove = None
def convert_word_to_glove(text):
    global glove
    if glove is None:
        glove = torchtext.vocab.GloVe(name="6B", dim=50)  # TODO tune number

    glove_text = [glove[x] for x in text]
    return glove_text

def load_data_and_convert_to_glove():
    df = pd.read_csv(github_data_clean_data)
    subdf = df[['data', 'label']]
    subdf['data'] = subdf['data'].apply(lambda x: convert_word_to_glove(x))

    converted_data = subdf[['data']].values
    label_numpy = subdf[['label']].values
    converted_label = [torch.from_numpy(label).squeeze(0) for label in label_numpy]

    return list(zip(converted_data, converted_label))


def split_data():
    pass



if __name__== "__main__":
    data_set = load_data_and_convert_to_glove()
    print(data_set[0])
    split_data()
