#!/usr/bin/env python

import gensim.downloader as api
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')


class GroundingModel(object):

    def __init__(self, known_words, embedding_model_dir="glove-wiki-gigaword-50",
                 threshold=0.7):

        self.known_words = known_words
        self.embedding_model = api.load(embedding_model_dir)
        self.stop_words = set(stopwords.words('english'))
        self.word_encoder = self._encode_known_words()
        self.threshold = threshold

    def _compute_multiple_words_vector(self, words):
        words_in_vocab = [word for word in words if word in self.embedding_model.vocab.keys()]
        return [np.sum(self.embedding_model[word] for word in words_in_vocab)][0]

    def _find_similar_word(self, w):
        possible_object = None
        max_cosine_sim = 0
        for word in self.word_encoder.keys():
            cosine_similarity = np.dot(w, self.word_encoder[word]) / (np.linalg.norm(w) * np.linalg.norm(
                self.word_encoder[word]))

            if cosine_similarity >= self.threshold and cosine_similarity >= max_cosine_sim:
                possible_object = word
                max_cosine_sim = cosine_similarity
        return (possible_object, max_cosine_sim)

    def _encode_known_words(self):
        return {word: self._compute_multiple_words_vector(
            word.split()) for word in self.known_words}

    def predict(self, item):
        words = [word for word in item.split() if word not in self.stop_words]
        w = self._compute_multiple_words_vector(words)
        return self._find_similar_word(w)


if __name__ == '__main__':
    ground_model = GroundingModel(known_words=["beer", "coke", "fridge", "laptop", "book"], threshold=0.7)

    while True:
        try:
            item = input("ITEM: ")
            print(ground_model.predict(item=item))
        except KeyboardInterrupt:
            continue
