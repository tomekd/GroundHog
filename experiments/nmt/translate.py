#!/usr/bin/env python

import argparse
import cPickle
import logging

import numpy

from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\

from experiments.nmt.numpy_compat import argpartition

logger = logging.getLogger(__name__)


def parse_input(state, word2idx, line, raise_unk=False, idx2word=None):
    unk_sym = state['unk_sym_source']
    null_sym = state['null_sym_source']
    seqin = line.split()
    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx, sx in enumerate(seqin):
        seq[idx] = word2idx.get(sx, unk_sym)
        if seq[idx] >= state['n_sym_source']:
            seq[idx] = unk_sym

    seq[-1] = null_sym
    return seq

class BeamSearch(object):

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, ignore_unk=False, minlen=1):
        c = self.comp_repr(seq)[0]
        states = map(lambda x: x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        num_levels = len(states)

        fin_trans = []
        fin_costs = []

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t: t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))
            log_probs = numpy.log(self.comp_next_probs(c, k, last_words, *states)[0])

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, self.unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:, self.eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states)

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            states = map(lambda x: x[indices], new_states)

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 500:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, False, minlen)
            else:
                logger.error("Translation failed")

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_trans, fin_costs


def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen


def translate(lm_model, seq, n_samples, beam_search=None, ignore_unk=False,
              normalize=False):
    sentences = []
    trans, costs = beam_search.search(seq, n_samples,
            ignore_unk=ignore_unk, minlen=len(seq) / 2)
    if normalize:
        counts = [len(s) for s in trans]
        costs = [co / cn for co, cn in zip(costs, counts)]
    for i in range(len(trans)):
        sen = indices_to_words(lm_model.word_indxs, trans[i])
        sentences.append(" ".join(sen))
    return sentences, costs, trans


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--beam-size",
            type=int, help="Beam size")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File to save translations in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("--model",
            help="Path to the model")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']),
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'], 'rb'))

    beam_search = BeamSearch(enc_dec)
    beam_search.compile()

    if args.source and args.trans:
        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for i, line in enumerate(fsrc):
            seqin = line.strip()
            seq = parse_input(state, indx_word, seqin)
            trans, costs, _ = translate(lm_model, seq, n_samples, beam_search,
                                        args.ignore_unk, args.normalize)
            best = numpy.argmin(costs)
            print >>ftrans, trans[best]
            total_cost += costs[best]
            if (i + 1) % 100 == 0:
                ftrans.flush()
        print "Total cost of the translations: {}".format(total_cost)

        fsrc.close()
        ftrans.close()
    else:
        while True:
            seqin = raw_input('Input Sequence: ')
            seq = parse_input(state, indx_word, seqin)

            translate(lm_model, seq, 1,
                    beam_search=beam_search,
                    ignore_unk=args.ignore_unk, normalize=args.normalize)

if __name__ == "__main__":
    main()
