github_cleaned_data = "../data/cleaned_data/dataset1.csv"
kaggle_cleaned_train_data = "../data/cleaned_data/dataset2.csv"
kaggle_cleaned_test_data = "../data/cleaned_data/dataset3.csv"
merged_cleaned_test_data = "../data/cleaned_data/Dataset_Merged.csv"
GLOVE_INPUT_SIZE = 50
FAST_TEXT_INPUT_SIZE = 300
CHAR_HIDDEN = 33#dataset1: 33#merged:3818
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')