#!/usr/bin/env python

import gensim.downloader as api
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')


class GroundingModel(object):

    def __init__(self, known_words, model_dir="glove-wiki-gigaword-50",
                 threshold=0.8, remove_stop_words=True, do_lower_case=True):

        self.do_lower_case = do_lower_case
        self.remove_stop_words = remove_stop_words
        self.known_words = known_words
        self.embedding_model = api.load(model_dir)

        # Stop words for removing from NER recognition (CHANGE TO INCORPORATE OUR OWN STOPWORDS IN THE ONTOLOGY!!)
        if self.remove_stop_words:
            self.stop_words = set(stopwords.words('english'))
            # Remove any known words from the stopwords
            [self.stop_words.remove(word) for word in self.known_words if word in self.stop_words]
            self.stop_words.add("please")

        self.word_encoder = self._encode_known_words()
        self.threshold = threshold

    def _compute_multiple_words_vector(self, words):

        if self.do_lower_case:
            words = [word.lower() for word in words]

        words_in_vocab = [word for word in words if word in self.embedding_model.vocab.keys()]

        if words_in_vocab:
            return [np.sum(self.embedding_model[word] for word in words_in_vocab)][0]
        else:
            return np.zeros(self.embedding_model.vector_size)

    def _find_similar_word(self, w):
        possible_object = None
        max_cosine_sim = 0
        for word in self.word_encoder.keys():
            cosine_similarity = np.dot(w, self.word_encoder[word]) / (np.linalg.norm(w) * np.linalg.norm(
                self.word_encoder[word]))
            try:
                if cosine_similarity >= self.threshold and cosine_similarity >= max_cosine_sim:
                    possible_object = word
                    max_cosine_sim = cosine_similarity
            except ValueError:
                raise ValueError("ERROR IN FIND SIMILAR WORD (GROUNDING) !!")
        return possible_object

    def _encode_known_words(self):
        return {word: self._compute_multiple_words_vector(
            word.split()) for word in self.known_words}

    def predict(self, item_words):
        if self.remove_stop_words:
            words = [word for word in item_words if word not in self.stop_words]
        else:
            words = item_words
        w = self._compute_multiple_words_vector(words)
        return self._find_similar_word(w)


if __name__ == '__main__':

    known_items = ["beer", "coke", "fridge", "laptop", "book"]

    ground_model = GroundingModel(known_words=known_items, threshold=0.7,
                                  model_dir="glove-wiki-gigaword-50",
                                  remove_stop_words=True)

    while True:
        try:
            item = input("ITEM: ")
            print(ground_model.predict(item_words=item))
        except KeyboardInterrupt:
            continue
