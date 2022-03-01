#!/usr/bin/env python3
"""
Class that loads and preps a dataset for machine translation.
Portugese-English translation dataset.
Approximately 50000 training examples, 1100 validation examples,
and 2000 test examples.
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    Loads and preps a dataset for machine translation
    """
    def __init__(self):
        """
        Class constructor
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)

        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        # Portuguese tokenizer created from the training set
        self.tokenizer_pt = tokenizer_pt
        # English tokenizer created from the training set
        self.tokenizer_en = tokenizer_en

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        Returns: tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = \
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        tokenizer_en = \
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data), target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en
