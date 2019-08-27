#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import json
import copy
import os

import nltk

import rospy
import rospkg

from mbot_nlu_pytorch.modeling import BertClassificationInference, BertNerInference
from mbot_nlu_pytorch.grounding import GroundingModel


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
            slot = copy.deepcopy(key)
            pred_obj = {key: {"value": [word], "confidence": None}}
            confidence = conf
        elif tag == "I":
            if pred_obj and slot in pred_obj.keys():
                pred_obj[slot]["value"].append(word)
                confidence *= conf
        elif tag == "O" and pred_obj:
            pred_obj[slot]["confidence"] = confidence
            predicts.append(pred_obj)
            pred_obj = None
            confidence = 0.0
    if pred_obj:
        pred_obj[slot]["confidence"] = confidence
        predicts.append(pred_obj)
    return predicts


class NLUModel(object):

    def __init__(self, ontology_dir, config):

        # need to feed a config with all the needed parameters as input !
        dtype_config        = config["dtype_model"]
        intent_config       = config["intent_model"]
        slotfiling_config   = config["slotfiling_model"]
        grounding_config    = config["grounding_model"]
        node_params         = config["node_params"]
        
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("mbot_nlu_pytorch")

        rospy.loginfo("Loading domain ontology")
        with open(ontology_dir, "r") as fp:
            ontology = json.load(fp)

        rospy.loginfo("Creating dtype model")
        self.dtype_model = BertClassificationInference(
            model_dir=os.path.join(pkg_path, dtype_config["model_dir"])
        )
        
        rospy.loginfo("Creating intent model")
        self.intent_model = BertClassificationInference(
            model_dir=os.path.join(pkg_path, intent_config["model_dir"])
        )
        
        rospy.loginfo("Creating slot filing model")
        self.slot_filing_model = BertNerInference(
            model_dir=os.path.join(pkg_path, slotfiling_config["model_dir"])
        )
        
        self.ground_model = None 
        if node_params["grounding"]:
            known_words = ontology["known_words"]
            known_words = [word.encode('utf-8') for word in known_words]
            rospy.loginfo("Creating grounding model")
            self.ground_model = GroundingModel(
                known_words=known_words,
                model_dir=grounding_config["model_dir"],
                threshold=grounding_config["threshold"],
                remove_stop_words=grounding_config["remove_stop_words"],
                do_lower_case=grounding_config["do_lower_case"]
            )

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
                        value = pred[slot]["value"]
                        conf = pred[slot]["confidence"]
                        true_value = None

                        if self.ground_model:
                            true_value = self.ground_model.predict(value)

                        if true_value:
                            d_act["known_args"][slot].append({true_value: conf})

                        else:
                            d_act["unknown_args"][slot].append({' '.join(value): conf})

                except ValueError:
                    raise ValueError("ERROR ON PROCESSING NER PREDICTIONS")

                except IndexError:
                    raise IndexError("ERROR ON INDEX")

            # change from <print> to <logging.debug>
            rospy.logdebug("SENTENCE: {}".format(sentence))
            rospy.logdebug(d_act)

            d_acts.append(d_act)

        return d_acts


if __name__ == '__main__':

    nltk.download('punkt')
    nltk.download('stopwords')

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

            if sentence[0] == 'Q':
                exit()

            d_act = nlu_model.predict(sentence)

            #print(json.dumps(d_act, indent=4))

        except KeyboardInterrupt:
            continue
