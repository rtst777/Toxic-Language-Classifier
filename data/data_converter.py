import pandas as pd
import re
import string
import nltk

github_data_raw_file_name = "github/labeled_data.csv"
github_data_cleaned_file_name = "cleaned_data/dataset1.csv"
# nltk.download()
stopword = nltk.corpus.stopwords.words('english')

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

def remove_empty_string_token(text):
    text = list(filter(None, text))
    return text

def lower_case_text(text):
    text = [x.lower() for x in text]
    return text

def convert_github_data():
    df = pd.read_csv(github_data_raw_file_name)
    subdf = df[['tweet', 'class']]
    subdf.columns = ['data', 'label'] # 0: hate speech, 1: offensive language, 2: neither

    # clean tweet data
    # https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis
    subdf['data'] = subdf['data'].apply(lambda x: remove_punct(x))
    subdf['data'] = subdf['data'].apply(lambda x: tokenization(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_stopwords(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_empty_string_token(x))
    subdf['data'] = subdf['data'].apply(lambda x: lower_case_text(x))

    # save cleaned data to file
    subdf.to_csv(path_or_buf=github_data_cleaned_file_name)


# TODO
def convert_kaggle_data():
    pass

if __name__== "__main__":
    convert_github_data()
    convert_kaggle_data()