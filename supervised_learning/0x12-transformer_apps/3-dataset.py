#!/usr/bin/env python3
"""
Class that loads and preps a dataset for machine translation.
Portugese-English translation dataset.
Approximately 50000 training examples, 1100 validation examples,
and 2000 test examples.
https://www.programmersought.com/article/38506277799/
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    Loads and preps a dataset for machine translation
    """
    def __init__(self, batch_size, max_len):
        """
        Class constructor
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.metadata = metadata

        self.data_train = examples['train']
        self.data_valid = examples['validation']

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

        def filter_max_length(x, y, max_length=max_len):
            """
            function for .filter() method
            """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        # Update data_train attribute
        # Filter out all examples that have either sentence
        # with more than max_len tokens
        self.data_train = self.data_train.filter(filter_max_length)
        # cache the dataset to increase performance
        self.data_train = self.data_train.cache()

        train_dataset_size = self.metadata.splits['train'].num_examples

        # shuffle the entire dataset
        self.data_train = self.data_train.shuffle(train_dataset_size)

        # split the dataset into padded batches of size batch_size
        padded_shapes = ([None], [None])
        self.data_train = \
            self.data_train.padded_batch(batch_size,
                                         padded_shapes=padded_shapes)

        # Prefetch the dataset using tf.data.experimental.AUTOTUNE
        # to increase performance
        self.data_train = \
            self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Update data_valid attribute
        # Filter out all examples that have either sentence
        # with more than max_len tokens
        self.data_valid = self.data_valid.filter(filter_max_length)

        # split the dataset into padded batches of size batch_size
        padded_shapes = ([None], [None])
        self.data_valid = \
            self.data_valid.padded_batch(batch_size,
                                         padded_shapes=padded_shapes)

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
