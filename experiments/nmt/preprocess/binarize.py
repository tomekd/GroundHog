#!/usr/bin/env python

import argparse
import cPickle
import logging
import os

import tables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('binarize')


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", type=argparse.FileType('r'),
                        help="the input file")
    parser.add_argument("-o", "--output",
                        help="the name of the pickled binarized text file")
    parser.add_argument("-v", "--vocab",
                        help="the vocabulary file")
    return parser.parse_args()


def safe_pickle(obj, filename):
    if os.path.isfile(filename) and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (filename))
    else:
        if os.path.isfile(filename):
            logger.info("Overwriting %s." % filename)
        else:
            logger.info("Saving to %s." % filename)
        with open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def safe_hdf(array, name):
    if os.path.isfile(name + '.hdf') and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (name + '.hdf'))
    else:
        if os.path.isfile(name + '.hdf'):
            logger.info("Overwriting %s." % (name + '.hdf'))
        else:
            logger.info("Saving to %s." % (name + '.hdf'))
        with tables.openFile(name + '.hdf', 'w') as f:
            atom = tables.Atom.from_dtype(array.dtype)
            filters = tables.Filters(complib='blosc', complevel=5)
            ds = f.createCArray(f.root, name.replace('.', ''), atom,
                                array.shape, filters=filters)
            ds[:] = array


def binarize(input_file, vocab):
    input_filename = os.path.basename(input_file.name)
    logger.info("Binarizing %s." % (input_filename))
    binarized_corpus = []
    for sentence_count, sentence in enumerate(input_file):
        words = sentence.strip().split(' ')
        binarized_sentence = [vocab.get(word, 1) for word in words]
        binarized_corpus.append(binarized_sentence)
    return binarized_corpus


if __name__ == "__main__":
    args = parse_args()
    with open(args.vocab) as vocab_file:
        vocab = cPickle.load(vocab_file)
    binarized_corpus = binarize(args.input, vocab)
    safe_pickle(binarized_corpus, args.output)
