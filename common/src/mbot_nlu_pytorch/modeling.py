from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification,
                                              BertForSequenceClassification)

import bert
from bert import tokenization

import nltk
from nltk import word_tokenize
nltk.download('punkt')


class BertNerModel(BertForTokenClassification):

    r"""

    Adaptation of BertForTokenClassification model.
    (https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py)

    Overwritten forward() method in order to only consider the valid
    tokens when computing the loss.

    DESCRIPTION

            **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
                Labels for computing the token classification loss.
                Indices should be in ``[0, ..., config.num_labels]``.
        Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification loss.
            **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
                Classification scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
                list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        Examples::
            >>> config = BertConfig.from_pretrained('bert-base-uncased')
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>>
            >>> model = BertForTokenClassification(config)
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
            >>> labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids, labels=labels)
            >>> loss, scores = outputs[:2]
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, attention_mask_label=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sequence_output, _ = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )

        # Checks valid tokens (no wordpiece tokens created in tokenization)
        # from (https://github.com/kamalkraj/BERT-NER/blob/experiment/run_ner.py)
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32, device=device
        )
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]

        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss

        return logits


class BertNerInference(object):

    def __init__(self, model_dir, device="cpu"):

        self.device = torch.device(device)
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        self.model.eval()

    def load_model(self, model_dir, model_config="model_config.json"):

        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        output_config_file = os.path.join(model_dir, CONFIG_NAME)
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertNerModel(
            config, num_labels=model_config["num_labels"]
        )
        model.load_state_dict(torch.load(output_model_file, map_location=self.device))
        model.to(self.device)
        #tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"], )
        vocab_path = os.path.join(model_dir, 'vocab.txt')
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=model_config["do_lower"]
        )
        return model, tokenizer, model_config

    def tokenize(self, text):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        tokens.insert(0, "[CLS]")
        valid_positions.insert(0, 1)
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids, input_mask, segment_ids, valid_positions

    def predict(self, text):
        input_ids, input_mask, segment_ids, valid_ids = self.preprocess(text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        valid_ids = torch.tensor([valid_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids, token_type_ids=segment_ids, attention_mask=input_mask, valid_ids=valid_ids
            )
        logits = F.softmax(logits, dim=2)
        logits_label = torch.argmax(logits, dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]
        logits_confidence = [values[label].item() for values, label in zip(logits[0], logits_label)]

        logits = []
        pos = 0
        for index, mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index - pos], logits_confidence[index - pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label], confidence) for label, confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [(word, {"tag": label, "confidence": confidence}) for word, (label, confidence) in zip(words, labels)]
        return output


class BertJointModel(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                ner_labels=None, classif_labels=None, valid_ids=None, attention_mask_label=None):

        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )

        # Checks valid tokens (no wordpiece tokens created in tokenization)
        # from (https://github.com/kamalkraj/BERT-NER/blob/experiment/run_ner.py)
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32
        )
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]

        sequence_output = self.dropout(valid_output)
        sequence_logits = self.classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        pooled_logits = self.classifier(pooled_output)

        if ner_labels is not None and classif_labels is not None:
            criteria = CrossEntropyLoss()
            # Only keep active parts of the loss for NER
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = sequence_logits.view(-1, self.num_labels)[active_loss]
                active_labels = ner_labels.view(-1)[active_loss]
                loss = criteria(active_logits, active_labels)
            else:
                # NEED TO ADD 2 NUM_LABELS !! HOW ??
                loss = criteria(sequence_logits.view(-1, self.num_labels), ner_labels.view(-1))

            loss += criteria(pooled_logits.view(-1, self.num_labels), classif_labels.view(-1))
            return loss

        return {"sequence_logits": sequence_logits, "pooled_logits": pooled_logits}


class BertClassificationModel(BertForSequenceClassification):

    r"""
            **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
                Labels for computing the sequence classification/regression loss.
                Indices should be in ``[0, ..., config.num_labels]``.
                If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
                If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification (or regression if config.num_labels==1) loss.
            **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
                list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        Examples::
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)
            loss, logits = outputs[:2]
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                criteria = MSELoss()
                loss = criteria(logits.view(-1), labels.view(-1))
            else:
                criteria = CrossEntropyLoss()
                loss = criteria(logits.view(-1, self.num_labels), labels.view(-1))
            return loss

        return logits


class BertClassificationInference(object):

    def __init__(self, model_dir, device='cpu'):

        self.device = torch.device(device)
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        self.model.eval()

    def load_model(self, model_dir, model_config="model_config.json"):
        model_config = os.path.join(model_dir, model_config)
        model_config = json.load(open(model_config))
        output_config_file = os.path.join(model_dir, CONFIG_NAME)
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertClassificationModel(config, num_labels=model_config["num_labels"])
        model.load_state_dict(torch.load(output_model_file, map_location=self.device))
        model.to(self.device)
        #tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"], do_lower_case=model_config["do_lower"])
        vocab_path = os.path.join(model_dir, 'vocab.txt')
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=model_config["do_lower"]
        )
        return model, tokenizer, model_config

    def tokenize(self, text):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = [self.tokenizer.tokenize(word) for word in words]
        return tokens[0]

    def preprocess(self, text):
        """ preprocess """
        tokens = self.tokenize(text)
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        return input_ids, input_mask, segment_ids

    def predict(self, text):
        input_ids, input_mask, segment_ids = self.preprocess(text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask)

        pred = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        pred_label = pred.detach().cpu().numpy().tolist()[0]
        confidence = F.softmax(logits, dim=1)[0][pred_label].item()

        return {"utterance": text, "label": self.label_map[pred_label], "confidence": confidence}
