#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import os

import rospy
import rospkg

from mbot_nlu_pytorch.nlu_pytorch_model import NLUModel


def main():

    rospack = rospkg.RosPack()
    pkg_dir = rospack.get_path("mbot_nlu_pytorch")
    dtype_model_dir = os.path.join(pkg_dir, "common/src/models/dtype")
    intent_model_dir = os.path.join(pkg_dir, "common/src/models/intent")
    slot_filing_model_dir = os.path.join(pkg_dir, "common/src/models/slotfiling")

    nlu_model = NLUModel(
        dtype_model_dir=dtype_model_dir,
        intent_model_dir=intent_model_dir,
        slot_filing_model_dir=slot_filing_model_dir
    )

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

