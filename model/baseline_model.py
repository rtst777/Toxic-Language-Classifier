from constants import github_cleaned_data
from constants import kaggle_cleaned_train_data
from constants import kaggle_cleaned_test_data
from collections import Counter
import pandas as pd
import numpy as np
from scipy.special import softmax

## Using naive-bayes as the baseline model
class ToxicBaseModel():
    def __init__(self):
        self.words_count_hs = {}
        self.words_count_ol = {}
        self.words_count_n = {}
        self.words_prob_hs = {}
        self.words_prob_ol = {}
        self.words_prob_n = {}
        self.prior = [1/3, 1/3, 1/3]


    def learn_distribution(self, file_paths):
        '''
        Learn the (Binomial) distribution
        :param file_paths: the paths of training dataset
        :return:
        '''
        self.words_count_hs, self.words_count_ol, self.words_count_n = get_words_counts(file_paths)
        sum_hs, sum_ol, sum_n = get_labels_sum(file_paths)

        # ML estimation with Laplace smoothing
        for word in self.words_count_hs.keys():
            self.words_prob_hs[word] = (self.words_count_hs[word] + 1) / (sum_hs + 2)
        for word in self.words_count_ol.keys():
            self.words_prob_ol[word] = (self.words_count_ol[word] + 1) / (sum_ol + 2)
        for word in self.words_count_n.keys():
            self.words_prob_n[word] = (self.words_count_n[word] + 1) / (sum_n + 2)

        return

    def classify_new_data(self, words):
        '''
        Use Naive Bayes classification to classify one new sentence
        :param words: one single sample (from the test set)
        :return: The classification result, and list of corresponding log posteriors
        '''
        hs_log_likelihood, ol_log_likelihood, n_log_likelihood = 0, 0, 0
        # log p(x, y=0)
        for word in self.words_prob_hs:
            if word in words:
                hs_log_likelihood += np.log(self.words_prob_hs[word])
            else:
                hs_log_likelihood += np.log(1 - self.words_prob_hs[word])
        hs_log_posterior = np.log(self.prior[0]) + hs_log_likelihood
        # log p(x, y=1)
        for word in self.words_prob_ol:
            if word in words:
                ol_log_likelihood += np.log(self.words_prob_ol[word])
            else:
                ol_log_likelihood += np.log(1 - self.words_prob_ol[word])
        ol_log_posterior = np.log(self.prior[1]) + ol_log_likelihood
        # log p(x, y=2)
        for word in self.words_prob_n:
            if word in words:
                n_log_likelihood += np.log(self.words_prob_n[word])
            else:
                n_log_likelihood += np.log(1 - self.words_prob_n[word])
        n_log_posterior = np.log(self.prior[2]) + n_log_likelihood

        # Use the softmax to find the prediction result
        log_posterior = [hs_log_posterior, ol_log_posterior, n_log_posterior]
        pred = np.argmax(softmax(np.asarray(log_posterior)))

        return pred, log_posterior


    def performance_measure(self, file, file_paths):

        # Store the classification results
        performance_measures = np.zeros([3, 3])
        correct = 0

        log_posteriors, true_indices = [], []
        df = pd.read_csv(file)
        df = df[:20000]
        for _, row in df.iterrows():
            pred, log_posterior = self.classify_new_data(row['data'])
            correct += (pred == row['label'])

        # The simplest way is to compute the test accuracy
        return float(correct)/len(df)

def get_words_counts(file_paths):
    """
    Returns dicts with the count of words with different labels
    """
    counts_hs, counts_ol, counts_n = Counter(), Counter(), Counter()
    for f in file_paths:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            if row['label'] == 0:
                for word in set(row['data']):
                    counts_hs[word] += 1
            elif row['label'] == 1:
                for word in set(row['data']):
                    counts_ol[word] += 1
            else:
                for word in set(row['data']):
                    counts_n[word] += 1

    return counts_hs, counts_ol, counts_n

def get_labels_sum(file_paths):
    '''
    Return the total number of data with one specific label
    '''

    sum_hs, sum_ol, sum_n = 0, 0, 0
    for f in file_paths:
        df = pd.read_csv(f)
        sum_hs += len(df[df['label'] == 0])
        sum_ol += len(df[df['label'] == 1])
        sum_n += len(df[df['label'] == 2])

    return sum_hs, sum_ol, sum_n


if __name__ == '__main__':
    file_paths = [github_cleaned_data, kaggle_cleaned_train_data]
    file = kaggle_cleaned_test_data
    toxic_base_model = ToxicBaseModel()
    toxic_base_model.learn_distribution(file_paths)
    print(toxic_base_model.performance_measure(file, file_paths))


