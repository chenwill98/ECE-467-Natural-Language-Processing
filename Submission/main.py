# William Chen
# Professor Sable
# ECE-467-1 Natural Language Processing
# Naive Bayes Model

import math
import nltk
import contractions
import string
import random
import re
import os
import numpy as np


# Splits the training set into a training and testing set, 75% and 25% respectively
def split_tuning_set(train_file_name, split_train_file_name, test_file_name):
    data = []
    with open(train_file_name, "r") as train_file:
        for line in train_file:
            data += [line.strip()]

    random.shuffle(data)

    train_data = data[:int(len(data) * 0.75)]
    test_data = data[int(len(data) * 0.75):]

    with open(split_train_file_name, "a") as split_train_file, \
            open(test_file_name + ".labels", "a") as split_test_file_labels, \
            open(test_file_name + ".list", "a") as split_test_file_list:
        for line in train_data:
            split_train_file.write(line)
            split_train_file.write("\n")

        for line in test_data:
            split_test_file_labels.write(line)
            split_test_file_labels.write("\n")
            split_test_file_list.write(line.split()[0])
            split_test_file_list.write("\n")


# Tokenizer that also cleans the tokens up with multiple options that can be changed for easy testing and comparison
def tokenize(document, remove_numbers=True, unhyphenated=True, lowercase=True, remove_punct=True, remove_stopwords=True,
             stemmer=True, lemmatizer=True):
    # Removes all of the numbers
    if remove_numbers:
        document_str = re.sub(r'\d+', '', document)
    # Spaces out hyphenated words so that it doesn't get removed when all tokens with punctuation is removed
    if unhyphenated:
        document_str.replace('-', ' ')
    # Splits all the words into tokens
    tokens = nltk.word_tokenize(document)
    # Sets all the tokens to lowercase, so that the model isn't case sensitive
    if lowercase:
        tokens = [token.lower() for token in tokens]
    # Removes all extraneous punctuation, with exceptions
    if remove_punct:
        tokens = [token for token in tokens if token.isalpha()]
    # Removes all of the stop words based on the NLTK corpus of stop words
    if remove_stopwords:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    # Applies porter stemming to all the words based on from the NLTK library
    if stemmer:
        porter = nltk.stem.porter.PorterStemmer()
        tokens = [porter.stem(token) for token in tokens]
    # Applies NLTK lemmatization to all the words, is mutually exclusive with stemmer
    if lemmatizer:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


# Takes in the path and category for the training documents and tokenizes it, while tallying up the statistics for
# Naive Bayes classification
def train(keys, corpus_data):
    path = keys[0]
    category = keys[1]

    with open(path, "r") as document:

        tokens = tokenize(document.read(), lemmatizer=False)

        if category not in corpus_data['category_list']:
            # Initialize the dict in the category list
            corpus_data['category_list'][category] = {}
            # Initialize the count for each category
            corpus_data['category_count'][category] = 1
        else:
            corpus_data['category_count'][category] += 1

        # Goes through all the tokens, adds them to the token list for each and to the vocabulary
        for token in tokens:
            if token not in corpus_data['category_list'][category]:
                corpus_data['category_list'][category][token] = 1
            else:
                corpus_data['category_list'][category][token] += 1
            # if token not in corpus_data['vocabulary']:
            corpus_data['vocabulary'][token] = 1

    return corpus_data

# Takes in the corpus data and the path for the file to classify, and runs Naive Bayes to return a prediction
def predict(key, corpus_data):
    # Initializes a dict for holding the probability values for each category
    category_prob = dict.fromkeys(corpus_data['category_list'].keys(), 0)
    laplace_smoothing = 0.1

    with open(key, "r") as document:

        tokens = tokenize(document.read(), lemmatizer=False)

        for category in corpus_data['category_list']:

            log_prior = np.log(corpus_data['category_count'][category] / sum(corpus_data['category_count'].values()))
            category_prob[category] += log_prior

            words_in_category = sum(corpus_data['category_list'][category].values())
            vocab_count = len(corpus_data['vocabulary'].keys()) * laplace_smoothing

            for token in tokens:
                if token in corpus_data['vocabulary']:
                    # Due to the nature of dicts, I have to make a condition for whether it exists or else it will
                    # throw an error
                    if token in corpus_data['category_list'][category]:
                        token_count = corpus_data['category_list'][category][token]
                        log_likelihood = np.log((token_count + laplace_smoothing) / (words_in_category + vocab_count))
                    else:
                        log_likelihood = np.log((0 + laplace_smoothing) / (words_in_category + vocab_count))
                else:
                    # Discard words not in the vocabulary
                    log_likelihood = 0
                category_prob[category] += log_likelihood

    prediction = max(category_prob, key=category_prob.get)

    return prediction


if __name__ == '__main__':
    train_file_name = input("Name of training file?")
    test_file_name = input("Name of testing file?")
    prediction_file_name = input("Name of prediction file?")

    # # Hardcoded for ease of testing
    # train_file_name = "corpus3_train.labels"
    # split_train_file_name = "corpus3_split_train.labels"
    # test_file_name = "corpus3_split_test"
    # prediction_file_name = "corpus3_predictions.labels"
    #
    # # Erases the previous prediction file for testing since the program writes by appending to it
    # if os.path.exists(split_train_file_name):
    #     os.remove(split_train_file_name)
    # if os.path.exists(test_file_name + ".list"):
    #     os.remove(test_file_name + ".list")
    # if os.path.exists(test_file_name + ".labels"):
    #     os.remove(test_file_name + ".labels")
    # if os.path.exists(prediction_file_name):
    #     os.remove(prediction_file_name)
    # split_tuning_set(train_file_name, split_train_file_name, test_file_name)
    #
    # train_file_name = split_train_file_name
    # test_file_name = test_file_name + ".list"

    # corpus_data tracks all the token counts for each category as well as a count of how many articles are
    # in each category and in total
    corpus_data = {'category_count': {},
                   'category_list': {},
                   'vocabulary': {}}

    # Opens the training data, processes it, and retrieves the populated corpus_data
    with open(train_file_name, "r") as train_file:
        for line in train_file:
            keys = line.split()
            corpus_data = train(keys, corpus_data)

    # Takes the corpus_data, makes predictions, and writes it to the prediction file
    with open(test_file_name, "r") as test_file, open(prediction_file_name, "a") as prediction_file:
        for line in test_file:
            key = line.strip()
            prediction = predict(key, corpus_data)
            prediction_file.write(key + " " + prediction)
            prediction_file.write("\n")
