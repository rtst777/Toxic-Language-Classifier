import pandas as pd
import re
import string
import nltk
import numpy as np

github_data_raw_file_name = "github/labeled_data.csv"
github_data_cleaned_file_name = "cleaned_data/dataset1.csv"
kaggle_data_raw_train_file_name = "kaggle/train.csv"
kaggle_data_cleaned_train_file_name = "cleaned_data/dataset2.csv"
kaggle_data_raw_test_data_file_name = "kaggle/test.csv"
kaggle_data_raw_test_label_file_name = "kaggle/test_labels.csv"
kaggle_data_cleaned_test_file_name = "cleaned_data/dataset3.csv"
# nltk.download()
stopword = nltk.corpus.stopwords.words('english')

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)  # TODO double check if it is necessary to remove number
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
def clean_unicode(text):
    return [x.encode('ascii', 'ignore') for x in text]
def convert_github_data():
    df = pd.read_csv(github_data_raw_file_name)
    subdf = df[['tweet', 'class']]
    subdf.columns = ['data', 'label'] # 0: hate speech: identity hate, threat, insult, , 1: offensive language: toxic,obscene, servere_toxic, 2: neither

    # clean tweet data
    # https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis
    subdf['data'] = subdf['data'].apply(lambda x: remove_punct(x))
    subdf['data'] = subdf['data'].apply(lambda x: tokenization(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_stopwords(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_empty_string_token(x))
    subdf['data'] = subdf['data'].apply(lambda x: lower_case_text(x))
    subdf['data'] = subdf['data'].apply(lambda x: clean_unicode(x))
    # save cleaned data to file
    subdf.to_csv(path_or_buf=github_data_cleaned_file_name, index=False)
    idx_list = np.random.randint(low=0, high=len(subdf) - 1, size=int(len(subdf) / div)).tolist()
    newdf = subdf.iloc[idx_list,:]
    newdf.to_csv(path_or_buf="cleaned_data/dataset4.csv", index=False)
    if merge:
        subdf.to_csv(path_or_buf="cleaned_data/Dataset_Merged.csv", index=False)

# TODO
def convert_kaggle_train_data():
    df = pd.read_csv(kaggle_data_raw_train_file_name)
    subdf = df[['comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']].copy()
    subdf.insert(7, "label", -1, allow_duplicates=False)
    for row in range(len(subdf)):
        if subdf.iloc[row]['toxic'] or subdf.iloc[row]['severe_toxic'] or subdf.iloc[row]['obscene']:
            subdf.set_value(row, 'label', 1)
        elif subdf.iloc[row]['threat'] or subdf.iloc[row]['insult'] or subdf.iloc[row]['identity_hate']:
            subdf.set_value(row, 'label', 0)
        else:
            subdf.set_value(row, 'label', 2)
    subdf = subdf[['comment_text','label']]
    subdf.columns = ['data', 'label']
    subdf['data'] = subdf['data'].apply(lambda x: remove_punct(x))
    subdf['data'] = subdf['data'].apply(lambda x: tokenization(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_stopwords(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_empty_string_token(x))
    subdf['data'] = subdf['data'].apply(lambda x: lower_case_text(x))
    subdf['data'] = subdf['data'].apply(lambda x: clean_unicode(x))
    idx_list = np.random.randint(low=0, high=len(subdf)-1, size=int(len(subdf)/div)).tolist()
    newdf = subdf.iloc[idx_list,:]
    newdf.to_csv(path_or_buf="cleaned_data/dataset4.csv", index=False)
    subdf.to_csv(path_or_buf=kaggle_data_cleaned_train_file_name, index=False)
    if merge:
        subdf.to_csv(path_or_buf="cleaned_data/Dataset_Merged.csv", mode='a', header=False,index=False)

def convert_kaggle_test_data():
    df_data = pd.read_csv(kaggle_data_raw_test_data_file_name)
    df_label = pd.read_csv(kaggle_data_raw_test_label_file_name)
    subdf = df_data[['comment_text']]
    subdf.insert(1, "label", -1, allow_duplicates=False)
    for row in range(len(df_data)):
        if df_label.iloc[row]['toxic'] or df_label.iloc[row]['severe_toxic'] or df_label.iloc[row]['obscene']:
            subdf.set_value(row, 'label', 1)
        elif df_label.iloc[row]['threat'] or df_label.iloc[row]['insult'] or df_label.iloc[row]['identity_hate']:
            subdf.set_value(row, 'label', 0)
        else:
            subdf.set_value(row, 'label', 2)
    subdf.columns = ['data', 'label']
    subdf['data'] = subdf['data'].apply(lambda x: remove_punct(x))
    subdf['data'] = subdf['data'].apply(lambda x: tokenization(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_stopwords(x))
    subdf['data'] = subdf['data'].apply(lambda x: remove_empty_string_token(x))
    subdf['data'] = subdf['data'].apply(lambda x: lower_case_text(x))
    subdf['data'] = subdf['data'].apply(lambda x: clean_unicode(x))
    idx_list = np.random.randint(low=0, high=len(subdf)-1, size=int(len(subdf)/div)).tolist()
    newdf = subdf.iloc[idx_list,:]
    newdf.to_csv(path_or_buf="cleaned_data/dataset4.csv", index=False)
    subdf.to_csv(path_or_buf=kaggle_data_cleaned_test_file_name, index=False)
    if merge:
        subdf.to_csv(path_or_buf="cleaned_data/Dataset_Merged.csv", mode='a', header=False, index=False)

if __name__== "__main__":
    global merge
    merge = False
    global div
    div = 25
    np.random.seed(seed=420)
    convert_github_data()
    convert_kaggle_train_data()
    convert_kaggle_test_data()