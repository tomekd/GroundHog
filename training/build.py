#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import subprocess
import logging
import ConfigParser
import shutil
from pprint import pprint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('build')


def print_configuration(config):
    """Print configuration file"""
    for section in config.sections():
        print('[{}]'.format(section))
        for key, value in config.items(section):
            print('{} = {}'.format(key, value))


def get_configuration():
    """Read configuration file"""
    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])
    working_dir = config.get('training', 'working_dir')
    try:
        os.mkdir(working_dir)
    except OSError:
        pass
    shutil.copy(sys.argv[1], working_dir + '/config.cfg')

    config.add_section('build')
    return config


def create_dirs(config):
    """ Create necessary directories. """
    logger.info("Creating directories.")
    working_dir = config.get('training', 'working_dir')
    for directory in (working_dir,
                      working_dir + '/vocab',
                      working_dir + '/ivocab',
                      working_dir + '/binarized'):
        try:
            os.mkdir(directory)
        except OSError:
            pass


def create_vocab(config):
    """Create vocabulary from corpus (WORD -> INDEX)."""
    logger.info("Creating vocabulary.")

    vocab_file = '{}/vocab/{}.pkl'.format(config.get('training', 'working_dir'),
                                          config.get('general', 'prefix'))
    config.set('build', 'vocab_file', vocab_file)
    vocab_size = config.get('corpus', 'vocab_size')
    input_corpus = '{}.{}'.format(config.get('corpus', 'corpus'),
                                  config.get('corpus', 'target_extension'))
    script_path = ''.join([config.get('general', 'groundhog_path'),
                          '/experiments/nmt/preprocess/make_dict.py'])
    command = ' '.join([script_path, '-o', vocab_file, '-s', vocab_size,
                        '-i', input_corpus])

    if not os.path.isfile(vocab_file):
        logger.info(command)
        subprocess.call(command, shell=True)
    else:
        logger.info("The file '{}' exists, using this one.".format(vocab_file))


def create_ivocab(config):
    """ Create the inverted vocab: INDEX ->WORD. """
    logger.info("Creating the inverted vocab.")

    ivocab_file = '{}/ivocab/{}.pkl'.format(
        config.get('training', 'working_dir'),
        config.get('general', 'prefix'))
    config.set('build', 'ivocab_file', ivocab_file)

    vocab_file = config.get('build', 'vocab_file')
    script_path = ''.join([config.get('general', 'groundhog_path'),
                          '/experiments/nmt/preprocess/invert-dict.py'])

    command = ' '.join(['python', script_path, vocab_file, ivocab_file])
    if not os.path.isfile(ivocab_file):
        logger.info(command)
        subprocess.call(command, shell=True)
    else:
        logger.info("The file '{}' exists, using this one.".format(ivocab_file))


def binarize(config, side):
    """ Binarize corpus """
    logger.info("Binarizing corpus.")

    vocab_file = config.get('build', 'vocab_file')

    input_corpus = '{}.{}'.format(config.get('corpus', 'corpus'),
                                  config.get('corpus', 'target_extension'))
    output_pkl = '{}/binarized/{}.{}.pkl'.format(
        config.get('training', 'working_dir'),
        config.get('general', 'prefix'),
        side)

    script_path = ''.join([config.get('general', 'groundhog_path'),
                          '/experiments/nmt/preprocess/binarize.py'])

    command = ' '.join([script_path,
                        '-i', input_corpus,
                        '-v', vocab_file,
                        '-o', output_pkl])
    if not os.path.isfile(output_pkl):
        logger.info(command)
        subprocess.call(command, shell=True)
    else:
        logger.info("The file '{}' exists, using this one.".format(output_pkl))

    logger.info("Converting PKL to HDF5.")

    convert_script_path = ''.join([config.get('general', 'groundhog_path'),
                                   '/experiments/nmt/preprocess/convert-pkl2hdf5.py'])

    output_hdf5 = '{}/binarized/{}.{}.h5'.format(
        config.get('training', 'working_dir'),
        config.get('general', 'prefix'),
        side)
    config.set('build', '{}_hdf5'.format(side), output_hdf5)

    command = ' '.join(['python', convert_script_path,
                        output_pkl, output_hdf5])
    if not os.path.isfile(output_hdf5):
        logger.info(command)
        subprocess.call(command, shell=True)
    else:
        logger.info('Using already existing {}.'.format(output_hdf5))


def shuffle_corpus(config):
    """ Shuffle parallel corpus. """
    logger.info("Shuffling corpus.")

    input_files = [config.get('build', '{}_hdf5'.format(side))
                   for side in ('source', 'target')]

    output_files = [name[:-3] + '_shuf.h5' for name in input_files]

    config.set('build', 'source_shuf', output_files[0])
    config.set('build', 'target_shuf', output_files[1])

    args = ' '.join(input_files + output_files)

    script_path = ''.join([config.get('general', 'groundhog_path'),
                          '/experiments/nmt/preprocess/shuffle-hdf5.py'])
    command = ' '.join(['python', script_path, args])

    if not os.path.isfile(output_files[0]) or not os.path.isfile(output_files[1]):
        subprocess.call(command, shell=True)
    else:
        print('Using existing files: {} and {}').format(*output_files)


def preprocess_data(config):
    """ Preprocess corpora. """

    create_dirs(config)
    create_vocab(config)
    create_ivocab(config)

    for side in ('source', 'target'):
        binarize(config, side)

    shuffle_corpus(config)


def update_path(path):
    items = path.split('/')
    return "./{}/{}".format(items[-2], items[-1])

def create_proto(config):
    """ Creating prototype. """
    sys.path.insert(1, config.get('general', 'groundhog_path') + '/experiments/nmt')
    import state as stt

    state = stt.prototype_search_state()

    prefix = config.get('general', 'prefix')
    working_dir = config.get('training', 'working_dir')


    state['source'] = [update_path(config.get('build', 'source_shuf'))]
    state['target'] = [update_path(config.get('build', 'target_shuf'))]

    state['indx_word'] = update_path(config.get('build', 'ivocab_file'))
    state['indx_word_target'] = update_path(config.get('build', 'ivocab_file'))

    state['word_indx'] = update_path(config.get('build', 'vocab_file'))
    state['word_indx_trgt'] = update_path(config.get('build', 'vocab_file'))

    state['null_sym_source'] = 0
    state['null_sym_target'] = 0
    state['n_sym_source'] = int(config.get('corpus', 'vocab_size'))
    state['n_sym_target'] = int(config.get('corpus', 'vocab_size'))
    state['seqlen'] = config.get('corpus', 'seqlen')

    for option in config.options('corpus'):
        if option in state:
            print "Changing option ", option, " parameters in state.py from config.cfg"
            print option, " = ", config.get('corpus', option)
            if config.get('corpus', option).isdigit():
                state[option] = config.getint('corpus', option)
            else:
                state[option] = config.get('corpus', option)

    pprint(state)

    with open(working_dir + '/{}.proto.pkl'.format(prefix), 'wb') as _file:
        pickle.dump(state, _file, 2)

    with open(working_dir + '/{}.proto.txt'.format(prefix), 'w') as _file:
        for key, value in state.iteritems():
            _file.write('{} = {}\n'.format(key, value))


def create_makefile(config):
    """ Create Makefile """
    prefix = config.get('general', 'prefix')
    working_dir = config.get('training', 'working_dir')
    device = config.get('training', 'device')
    moses_path = config.get('general', 'moses_dir')
    test_set = config.get('evaluation', 'input')
    reference = config.get('evaluation', 'reference')
    state = './{}.proto.pkl'.format(prefix)
    model = './{}_model.npz'.format(prefix)
    groundhog_path = config.get('general', 'groundhog_path')
    template = """#!/usr/bin/bash
THEANO_FLAGS='floatX=float32,device={},nvcc.fastmath=True,mode=FAST_RUN'
MOSES_PATH={}
STATE={}
MODEL={}
GROUNDHOG_PATH={}
TEST_IN={}
REFERENCE={}
TEST_OUT=translated.txt

train:
\tPYTHONPATH=$(GROUNDHOG_PATH) THEANO_FLAGS=$(THEANO_FLAGS) python $(GROUNDHOG_PATH)/experiments/nmt/train.py --state $(STATE)
.PHONY: train

test:
\tPYTHONPATH=$(GROUNDHOG_PATH) THEANO_FLAGS=$(THEANO_FLAGS) python $(GROUNDHOG_PATH)/experiments/nmt/sample.py \
\t--state $(STATE) --beam-search --beam-size 50 --ignore-unk \
\t--source $(TEST_IN) --translate $(TEST_OUT) $(MODEL)

bleu:
\t$(MOSES_PATH)/scripts/generic/multi-blue.perl -lc $(REFERENCE) < $(TEST_OUT) | tee bleu.txt


""".format(device, moses_path, state, model, groundhog_path, test_set,
           reference)
    with open('{}/Makefile'.format(working_dir), 'w') as makefile:
        makefile.write(template)


def main():
    """ main """
    config = get_configuration()
    # print_configuration(config)
    preprocess_data(config)
    create_proto(config)
    create_makefile(config)


if __name__ == "__main__":
    main()
