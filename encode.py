from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np


def encode_to_onehot(train_x):
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform(train_x)
    feature_name = vectorizer.get_feature_names()
    vabulary = vectorizer.vocabulary_
    train_x = count.toarray()
    train_x = np.array(train_x)
    return train_x


def encode_to_zeor_one(train_y):
    for i in range(train_y.shape[0]):
        if train_y[i] == "Negative":
            train_y[i] = 0
        else:
            train_y[i] = 1
    return train_y


def decode_to_emotion(result):
    buffer = []
    for i in range(result.shape[0]):
        if result[i] <= 0.5:
            buffer.append("Negative")
        else:
            buffer.append("Positive")
    return buffer
