#!/usr/bin/env python3
"""
Performs semantic search on a corpus of documents.
https://tfhub.dev/google/universal-sentence-encoder-large/5
"""
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Returns: the reference text of the document most similar to 'sentence'
    """
    documents = [sentence]

    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(corpus_path + '/' + filename,
                      mode='r', encoding='utf-8') as file:
                documents.append(file.read())

    # Load model that encodes text into 512 dimensional vectors
    embed = \
        hub.load("https://tfhub.dev/google/" +
                 "universal-sentence-encoder-large/5")

    # sentence + 91 documents
    # (92, 512)
    embeddings = embed(documents)
    # The semantic similarity of two sentences is
    # the inner product of the encodings.
    # (92, 92) Correlation matrix
    corr = np.inner(embeddings, embeddings)
    # most similar excluding itself
    most_similar = np.argmax(corr[0, 1:])
    # Add 1 because of the above line
    text = documents[most_similar + 1]

    return text
