#!/usr/bin/env python3
"""
Converts a gensim word2vec model to a keras Embedding layer
"""
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """
    Returns: the trainable keras Embedding
    """
    layer = model.wv.get_keras_embedding()

    return layer
