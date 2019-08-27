#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import copy
import json
import os
import string
import yaml

import numpy as np

import rospy
import rospkg

from mbot_nlu_pytorch.nlu_pytorch_model import NLUModel
from mbot_nlu_pytorch.msg import (InformSlot, DialogAct,
								DialogActArray, ASRHypothesis,
								ASRNBestList)


"""
Description: This function helps logging parameters as debug verbosity messages.

Inputs:
	- param_dict: a dictionary whose keys are the parameter's names and values the parameter's values.
"""
def logdebug_param(param_dict):
	[ rospy.logdebug( '{:<25}\t{}'.format(param[0], param[1]) ) for param in param_dict.items() ]


class NLUNode(object):

	def __init__(self, debug=False):

		# need to download a .yaml config with all the needed parameters !
		rospy.loginfo("Loading node config")
		config_path = rospy.myargv()[1]
		config = yaml.load(open(config_path))

		# get node params from config
		rate 			= config["node_params"]["rate"]
		node_name 		= config["node_params"]["name"]
		debug 			= config["node_params"]["debug"]
		ontology_dir   	= config["node_params"]["ontology_dir"]
		n_best_topic	= config["node_params"]["n_best_topic"]
		d_acts_topic	= config["node_params"]["d_acts_topic"]

		# initializes the node (if debug, initializes in debug mode)
		if debug == True:
			rospy.init_node(node_name, anonymous=False, log_level=rospy.DEBUG)
			rospy.loginfo("%s node created [DEBUG MODE]" % node_name)
		else:
			rospy.init_node(node_name, anonymous=False)
			rospy.loginfo("%s node created" % node_name)

		rospack = rospkg.RosPack()
		pkg_path = rospack.get_path("mbot_nlu_pytorch")
		ontology_dir = os.path.join(pkg_path, ontology_dir)

		rospy.loginfo('Creating NLU model')
		self.nlu_object = NLUModel(
			config = config,
			ontology_dir=ontology_dir
		)
		rospy.loginfo('NLU model created')


		self.nlu_request_received = False
		self.asr_n_best_list = None
		self.loop_rate = rospy.Rate(rate)

		rospy.Subscriber(n_best_topic, ASRNBestList, self.speechRecognitionCallback, queue_size=1)
		rospy.loginfo("Subscribed to topic <%s>", n_best_topic)

		self.pub_sentence_recog = rospy.Publisher(d_acts_topic, DialogActArray, queue_size=1)
		rospy.loginfo("Publishing to topic <%s>", d_acts_topic)

		rospy.loginfo("Node <%s> initialization completed! Ready to accept requests" % node_name)
		
	def speechRecognitionCallback(self, msg):

		rospy.loginfo('[Message received]')
		rospy.logdebug('{}'.format(msg))

		self.asr_n_best_list = msg
		self.nlu_request_received = True

	def preprocess_sentences(self, sentences):

		sentences = [
			' '.join(
				x for x in sentence.split() if x not in string.punctuation
			)
		for sentence in sentences]

		sentences = [
			' '.join(x.lower() for x in sentence.split())
		for sentence in sentences]

		sentences = [
			sentence.replace('[^\w\s]','')
		for sentence in sentences]

		sentences = [
			' '.join(
				x for x in sentence.split() if  not x.isdigit()
			)
		for sentence in sentences]

		return sentences

	def begin(self):

		while not rospy.is_shutdown():

			if self.nlu_request_received == True:

				rospy.loginfo('[Handling message]' )
				self.nlu_request_received = False

				pred_sentences = [hypothesis.transcript for hypothesis in self.asr_n_best_list.hypothesis]
				# preprocess user utterance hypothesis before feeding the semantic decoder
				pred_sentences = self.preprocess_sentences(pred_sentences)
				# compute probability distribution of each hypothesis through softmax of confidence scores
				confs = [hypothesis.confidence for hypothesis in self.asr_n_best_list.hypothesis]
				probs = np.exp(confs) / np.sum(np.exp(confs))

				rospy.logdebug('sentences to predict: {}'.format(pred_sentences))

				rospy.loginfo('Beginning Predictions!')
				preds = self.nlu_object.predict(pred_sentences)

				#  Creates DialogActs to publish
				dialogue_act_array_msg = DialogActArray()
				for i, pred in enumerate(preds):
					dialogue_act_msg = DialogAct()
					#  Adds d-type
					dtype = list(pred['d-type'].keys())[0]
					dtype_conf = pred['d-type'][dtype]
					dialogue_act_msg.dtype = dtype
					dialogue_act_msg.d_type_confidence = dtype_conf*probs[i]

					#  Adds intent
					if pred['intent']:
						intent = list(pred['intent'].keys())[0]
						intent_conf = pred['intent'][intent]
						dialogue_act_msg.slots.append(
							InformSlot(slot="intent", value=intent, confidence=intent_conf*probs[i], known=True)
						)

					#  Adds known slots
					for slot in list(pred["known_args"].keys()):
						value_list = pred["known_args"][slot]
						if value_list:
							for value_dict in value_list:
								value = list(value_dict.keys())[0]
								conf = value_dict[value]
								dialogue_act_msg.slots.append(
									InformSlot(slot=slot, value=value, confidence=conf*probs[i], known=True)
								)
					#  Adds unknown slots
					for slot in list(pred["unknown_args"].keys()):
						value_list = pred["unknown_args"][slot]
						if value_list:
							for value_dict in value_list:
								value = list(value_dict.keys())[0]
								conf = value_dict[value]
								dialogue_act_msg.slots.append(
									InformSlot(slot=slot, value=value, confidence=conf*probs[i], known=False)
								)
					
					dialogue_act_array_msg.dialogue_acts.append(copy.deepcopy(dialogue_act_msg))
				
				self.pub_sentence_recog.publish(dialogue_act_array_msg)

			self.loop_rate.sleep()


def main():

	nlu_node = NLUNode()
	nlu_node.begin()