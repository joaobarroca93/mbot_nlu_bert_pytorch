#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import json
#import ipdb

#from mbot_nlu_pytoch.modeling import BertClassificationInference, BertNerInference
#from mbot_nlu_pytoch.grounding import GroundingModel

from modeling import BertClassificationInference, BertNerInference
from grounding import GroundingModel


def reset_d_act():
    return {
        "d-type": None,
        "intent": None,
        "known_args": {"object": [], "person": [], "destination": [], "source": [], "what_to_tell": []},
        "unknown_args": {"object": [], "person": [], "destination": [], "source": [], "what_to_tell": []}
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

    def __init__(self, dtype_model_dir, intent_model_dir, slot_filing_model_dir,
                 ontology_dir, grounding_model_dir=None):

        with open(ontology_dir, "r") as fp:
            ontology = json.load(fp)

        self.dtype_model = BertClassificationInference(model_dir=dtype_model_dir)
        self.intent_model = BertClassificationInference(model_dir=intent_model_dir)
        self.slot_filing_model = BertNerInference(model_dir=slot_filing_model_dir)

        if grounding_model_dir:
            self.ground_model = GroundingModel(known_words=ontology["known_words"], model_dir=grounding_model_dir)

    def predict(self, sentences):

        # If it is only one sentence, wrap it in a list for prediction loop
        if isinstance(sentences, str):
            sentences = [sentences]

        # Prediction Loop
        d_acts = []
        for sentence in sentences:
            d_act = reset_d_act()

            dtype_pred = self.dtype_model.predict(sentence)
            intent_pred = self.intent_model.predict(sentence)
            slot_filing_pred = self.slot_filing_model.predict(sentence)
            preds = process_ner_pred(slot_filing_pred)

            d_act["d-type"] = {dtype_pred["label"]: dtype_pred["confidence"]}

            if dtype_pred["label"] == "inform":

                d_act["intent"] = {intent_pred["label"]: intent_pred["confidence"]}

                try:
                    for pred in preds:
                        slot = list(pred.keys())[0]
                        value = list(pred.values())[0]
                        conf = list(pred.values())[1]
                        true_value = None

                        if self.ground_model:
                            true_value = self.ground_model.predict(value)

                        if true_value:
                            d_act["known_args"][slot].append({true_value: conf})

                        else:
                            d_act["unknown_args"][slot].append({' '.join(value): conf})

                except ValueError:
                    #ipdb.set_trace()
                    raise ValueError("ERROR ON PROCESSING NER PREDICTIONS")

                except IndexError:
                    #ipdb.set_trace()
                    raise IndexError("ERROR ON INDEX")

            # change from <print> to <logging.debug>
            print(json.dumps(d_act, indent=4))

            d_acts.append(d_act)

        return d_acts


if __name__ == '__main__':

    nlu_model = NLUModel(
        dtype_model_dir="../model/dtype",
        intent_model_dir="../model/intent",
        slot_filing_model_dir="../model/slotfiling",
        grounding_model_dir="glove-wiki-gigaword-50",
        ontology_dir="../model/ontology.json"
    )

    while True:
        try:
            print("=" * 20)
            sentence = raw_input("SENTENCE: ")

            d_act = nlu_model.predict(sentence)

            #print(json.dumps(d_act, indent=4))

        except KeyboardInterrupt:
            continue
