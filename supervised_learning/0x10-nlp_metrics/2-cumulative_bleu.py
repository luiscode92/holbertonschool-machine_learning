#!/usr/bin/env python3
"""
Calculates the cumulative n-gram BLEU score for a sentence
"""
import numpy as np


def grams(sentence, n):
    """
    Creates grams
    """
    new = []
    len_sentence = len(sentence)

    for i, word in enumerate(sentence):
        s = word
        counter = 0
        j = 0

        for j in range(1, n):
            if len_sentence > i + j:
                s += " " + sentence[i + j]
                counter += 1
        if counter == j:
            new.append(s)

    return new


def transform_grams(references, sentence, n):
    """
    Returns: new_ref, new_sentence
    """
    if n == 1:
        return references, sentence

    new_sentence = grams(sentence, n)

    new_ref = []

    for ref in references:
        new_r = grams(ref, n)
        new_ref.append(new_r)

    return new_ref, new_sentence


def calc_precision(references, sentence, n):
    """
    Returns: precision
    """
    references, sentence = transform_grams(references, sentence, n)
    sentence_dict = {x: sentence.count(x) for x in sentence}
    references_dict = {}

    for ref in references:
        for gram in ref:
            if gram not in references_dict.keys() \
                    or references_dict[gram] < ref.count(gram):
                references_dict[gram] = ref.count(gram)

    appearances = {x: 0 for x in sentence}

    for ref in references:
        for gram in appearances.keys():
            if gram in ref:
                appearances[gram] = sentence_dict[gram]

    for gram in appearances.keys():
        if gram in references_dict.keys():
            appearances[gram] = min(references_dict[gram], appearances[gram])

    len_trans = len(sentence)
    precision = sum(appearances.values()) / len_trans

    return precision


def cumulative_bleu(references, sentence, n):
    """
    Returns: the cumulative n-gram BLEU score
    """
    precisions = [0] * n

    for i in range(0, n):
        precisions[i] = calc_precision(references, sentence, i + 1)

    geo_mean = np.exp(np.sum((1 / n) * np.log(precisions)))

    len_trans = len(sentence)

    closest_ref_idx = np.argmin([abs(len(x) - len_trans) for x in references])

    reference_length = len(references[closest_ref_idx])

    if len_trans > reference_length:
        BP = 1
    else:
        BP = np.exp(1 - float(reference_length) / len_trans)

    BLEU_score = BP * geo_mean

    return BLEU_score
