#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import json

import nltk
nltk.download('punkt')

from mbot_nlu_pytorch.bert import Ner


class NLUModel(object):

    def __init__(self, model_dir):

        self.model = Ner(model_dir)


    def predict(sentences):

        preds = None

        for sentence in sentences:

            output = model.predict(sentence)

            objs = []
            obj_object = []
            for word_tag_pair in output:
                word = word_tag_pair[0]
                tag = word_tag_pair[1]["tag"]
                prob = word_tag_pair[1]["confidence"]
                if tag == "Bobject":
                    obj_object.append(word)
                elif tag == "Iobject":
                    obj_object.append(word)
                else:
                    if obj_object:
                        objs.append(' '.join(obj_object))
                        obj_object = []
            if obj_object:
                objs.append(' '.join(obj_object))
                obj_object = []

            print("\n| Order: ", sentence)
            for o in objs:
                w = compute_multiple_words_vector(o.split(), embedding_model)
                true_obj = find_similar_word(w, word_encoder=items_encoder, threshold=threshold)
                if true_obj:
                    print("--> Known Item: ", true_obj)
                else:
                    print("--> Unknown Item: ", o)
                    # when I dont know an item I can ask for confirmation and then add it to the know items
                    #new_dict = {o: compute_multiple_words_vector(o.split(), embedding_model)}
                    #items_encoder.update(new_dict)

        return preds