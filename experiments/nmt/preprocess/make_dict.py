#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle
import logging
import os

from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('make-dict')


def safe_pickle(obj, filename, overwrite=False):
    if os.path.isfile(filename) and overwrite:
        logger.warning("Not saving %s, already exists." % (filename))
    else:
        if os.path.isfile(filename):
            logger.info("Overwriting %s." % filename)
        else:
            logger.info("Saving to %s." % filename)
        with open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def create_dictionary(input_filename, size):
    print input_filename
    counter = Counter()

    logger.info("Counting words in %s" % input_filename)
    counter = Counter()
    sentence_count = 0
    for line in input_filename:
        words = line.strip().split(' ')
        counter.update(words)
        sentence_count += 1
    logger.info("%d unique words in %d sentences with a total of %d words."
                % (len(counter), sentence_count, sum(counter.values())))
    vocab_count = counter.most_common(size - 2)
    logger.info("Creating dictionary of %s most common words, covering "
                "%2.1f%% of the text."
                % (size,
                    100.0 * sum([count for word, count in vocab_count]) /
                    sum(counter.values())))
    vocab = {'UNK': 1, '<s>': 0, '</s>': 0}
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 2
    return vocab


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", type=argparse.FileType('r'),
                        help="The input files")
    parser.add_argument("-s", "--size", type=int, metavar="N",
                        help="limit vocabulary size to this number.")
    parser.add_argument("-o", "--output",
                        help="The output file to write PKL file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vocab = create_dictionary(args.input, args.size)
    safe_pickle(vocab, args.output)
