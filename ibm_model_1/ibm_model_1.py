#!/usr/bin/env python3
#coding: utf-8

import sys
import argparse
import numpy as np
import spacy_udpipe
import tqdm

def load_input(args):
    if args.lemmatize:
        spacy_udpipe.download("en")
        spacy_udpipe.download("cs")
        en_nlp = spacy_udpipe.load("en")
        cs_nlp = spacy_udpipe.load("cs")
    en_vocab, cs_vocab = set(), set()
    en_sentences, cs_sentences = [], []
    with open(args.input, 'r') as input:
        lines = input.readlines()
        for line in tqdm.tqdm(lines, total=len(lines)):
            if args.lowercase:
                line = line.lower()
            en, cs, _, _ = line.split('\t')
            if args.lemmatize:
                cs = [token.lemma_ for token in cs_nlp(cs)]
                en = [token.lemma_ for token in en_nlp(en)]
            else:
                en, cs = en.split(), cs.split()
            en_vocab.update(en)
            cs_vocab.update(cs)
            en_sentences.append(en)
            cs_sentences.append(cs)
    en_vocab = dict([(w, idx) for idx, w in enumerate(list(en_vocab))])
    cs_vocab = dict([(w, idx + 1) for idx, w in enumerate(list(cs_vocab))])
    cs_vocab['<NULL>'] = 0
    en2id = lambda s: list(map(lambda w: en_vocab[w], s))
    cs2id = lambda s: [0,] + list(map(lambda w: cs_vocab[w], s))
    cs_sentences = list(map(cs2id, cs_sentences))
    en_sentences = list(map(en2id, en_sentences))
    return en_sentences, cs_sentences, en_vocab, cs_vocab

def train(en_sentences, f_sentences, len_en, len_f, max_iter = 10):
    t = np.zeros((len_en, len_f)) + 1 / len_en

    for idx in range(max_iter):
        sys.stderr.write(f'Iterration {idx}\n')
        count = np.zeros_like(t)
        total = np.zeros(len_f)
        for e_s, f_s in tqdm.tqdm(zip(en_sentences, f_sentences), total=len(en_sentences)):
            total_s = np.zeros(len_en)
            for e in e_s:
                total_s[e] += np.sum(t[e, f_s])
            for e in e_s:
                for f in f_s:
                    count[e, f] += t[e, f] / total_s[e]
                    total[f] += t[e, f] / total_s[e]
        nt = count / total
        
        changed = np.sum(np.argmax(nt, axis=0) != np.argmax(t, axis=0))
        loss = np.sum((t - nt) ** 2)
        sys.stderr.write(f'\tLoss:    {loss}\n')
        sys.stderr.write(f'\tChanges: {changed}\n')
        t = nt

    return t
    
def align(model, en_sentences, f_sentences, en, f):
    for e_s, f_s in zip(en_sentences, f_sentences):
        for idx, e in enumerate(e_s):
            a = np.argmax(model[e, f_s])
            if a > 0:
                sys.stdout.write(f'{idx + 1}-{a} ')
        sys.stdout.write('\t')
        for idx, e in enumerate(e_s):
            a = np.argmax(model[e, f_s])
            p = np.max(model[e, f_s])
            if a > 0:
                sys.stdout.write(f'{en[e]}-{f[f_s[a]]}-{p} ')
        sys.stdout.write('\n')


def main(args):
    pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--lowercase', action='store_true', default=False)
    parser.add_argument('--lemmatize', action='store_true', default=False)
    parser.add_argument('--iterations', type=int, default=7)

    parser.add_argument('input', type=str)
    args = parser.parse_args()

    en, cs, en_vocab, cs_vocab = load_input(args)
    model = train(en, cs, len(en_vocab), len(cs_vocab), args.iterations)
    flip = lambda d: {v: k for k, v in d.items()}
    align(model, en, cs, flip(en_vocab), flip(cs_vocab))


    main(args)