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

        # tokenizing the examples
        # Dataset.map Maps map_func across the elements of this dataset.
        self.data_train = self.data_train.map(self.tf_encode)

        # tokenizing the examples
        self.data_valid = self.data_valid.map(self.tf_encode)

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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.
        Returns: pt_tokens, en_tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        tf wrapper for the 'encode' instance method to be used with map()
        """
        result_pt, result_en = tf.py_function(func=self.encode, inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])
        # None allows any value
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
