#!/usr/bin/env python3
"""
Answers questions from a reference text
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Returns: a string containing the answer.
    If no answer is found, return None.
    """
    tokenizer = \
        BertTokenizer.from_pretrained('bert-large-uncased' +
                                      '-whole-word-masking-finetuned-squad')

    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_tokens = tokenizer.tokenize(question)

    reference_tokens = tokenizer.tokenize(reference)

    tokens = \
        ['[CLS]'] +\
        question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_word_ids)

    input_type_ids = \
        [0] * (1 + len(question_tokens) + 1) +\
        [1] * (len(reference_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1

    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[short_start: short_end + 1]

    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # 0-main.py
    # print(question_answer('Not a valid question?', reference))
    # output: None
    if not answer or question in answer:
        return None
    return answer


def answer_loop(reference):
    """
    Answers questions from a reference text
    """
    while True:
        question = input("Q: ").lower()

        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        else:
            answer = question_answer(question, reference)
            if answer is None:
                print("A: Sorry, I do not understand your question.")
            else:
                print("A:", answer)
