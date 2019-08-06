#!/usr/bin/env python

import os

import rospy
import rospkg

from mbot_nlu_pytorch.nlu_pytorch_model import NLUModel


def main():

    rospack = rospkg.RosPack()
    pkg_dir = rospack.get_path("mbot_nlu_pytorch")
    model_dir = os.path.join(pkg_dir, "common/src/model")

    nlu_model = NLUModel(model_dir=model_dir)

    sentences = [
        "I want a coffee",
        "bring me a beer",
        "I would like a coke",
        "a cup of tea",
        "I want a breakfast wrap and a coffee please",
        "can you bring me a chicken salad sandwich please",
        "I want a toast with ham and cheese",
        "I want the innocent mango",
        "I want a smoothie",
        "I want an italian biscotti",
        "I want a large cup of coffee",
        "I want a medium cup of tea",
        "I want a small cup",
        "I want some crisps",
        "I want a veggie pot",
        "bring me a water"
    ]

    nlu_model.predict(sentences)

