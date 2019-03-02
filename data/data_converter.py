import pandas as pd

github_data_raw_file_name = "github/labeled_data.csv"
github_data_cleaned_file_name = "cleaned_data/dataset1.csv"

def one_hot_encoding(dataframe, labels):
    for target_label_index in range(len(labels)):
        for i in range(len(labels)):
            label = labels[i]
            if i == target_label_index:
                dataframe.loc[dataframe['class'] == target_label_index, label] = 1
            else:
                dataframe.loc[dataframe['class'] == target_label_index, label] = 0

def convert_github_data():
    df = pd.read_csv(github_data_raw_file_name)
    subdf = df[['hate_speech', 'offensive_language', 'neither', 'class', 'tweet']]
    labels = ['hate_speech', 'offensive_language', 'neither']
    one_hot_encoding(subdf, labels)
    subdf[['hate_speech', 'offensive_language', 'neither', 'tweet']].to_csv(path_or_buf=github_data_cleaned_file_name)

# TODO
def convert_kaggle_data():
    pass

if __name__== "__main__":
    convert_github_data()
    convert_kaggle_data()