#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import numpy as np
import json

#import nltk
#nltk.download('punkt')

from mbot_nlu_pytorch.modeling import BertClassificationInference, BertNerInference


def reset_d_act():
    return {
        "d-type": None,
        "intent": None,
        "args": {"object": [], "person": [], "destination": [], "source": [], "what_to_tell": []}
    }


def process_ner_pred(ner_preds):
    predicts = []
    pred_obj = None
    confidence = 0.0
    for pred in ner_preds:
        word = pred[0]
        conf = pred[1]["confidence"]
        tag = pred[1]["tag"][0]
        key = pred[1]["tag"][1:]
        if tag == "B":
            pred_obj = {key: [word], "confidence": None}
            confidence = conf
        elif tag == "I":
            if pred_obj and key in pred_obj.keys():
                pred_obj[key].append(word)
                confidence *= conf
        elif tag == "O" and pred_obj:
            pred_obj["confidence"] = confidence
            predicts.append(pred_obj)
            pred_obj = None
            confidence = 0.0
    if pred_obj:
        pred_obj["confidence"] = confidence
        predicts.append(pred_obj)
    return predicts


class NLUModel(object):

    def __init__(self, dtype_model_dir, intent_model_dir, slot_filing_model_dir):

        self.dtype_model = BertClassificationInference(model_dir=dtype_model_dir)
        self.intent_model = BertClassificationInference(model_dir=intent_model_dir)
        self.slot_filing_model = BertNerInference(model_dir=slot_filing_model_dir)

    def predict(self, sentences):

        d_acts = []
        for sentence in sentences:
            dtype_pred = self.dtype_model.predict(sentence)
            intent_pred = self.intent_model.predict(sentence)
            slot_filing_pred = self.slot_filing_model.predict(sentence)
            preds = process_ner_pred(slot_filing_pred)

            d_act["d-type"] = {dtype_pred["label"]: dtype_pred["confidence"]}

            if dtype_pred["label"] == "inform":

                d_act["intent"] = {intent_pred["label"]: intent_pred["confidence"]}

                for pred in preds:
                    slot = list(pred.keys())[0]
                    value = list(pred.values())[0]
                    conf = list(pred.values())[1]
                    d_act["args"][slot].append({' '.join(value): conf})

            # change from <print> to <logging.debug>
            print("SENTENCE: ", sentence)
            print(json.dumps(d_act, indent=4))

            d_acts.append(d_act)
            d_act = reset_d_act()

        return d_acts
