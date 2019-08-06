#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import numpy as np
import json

import nltk
nltk.download('punkt')

from mbot_nlu_pytorch.bert_model import Ner


class NLUModel(object):

    def __init__(self, model_dir):

        self.model = Ner(model_dir)


    def predict(self, sentences):

        preds = None

        for sentence in sentences:

            output = self.model.predict(sentence)

            objs = []
            obj_object = []
            for word_tag_pair in output:
                word = word_tag_pair["word"]
                tag = word_tag_pair["tag"].encode(encoding="utf-8")
                prob = word_tag_pair["confidence"]
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
                print("--> Item: ", o)

        return preds