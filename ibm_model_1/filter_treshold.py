#!/usr/bin/env python3
#coding: utf-8

"""
Reads alignment from  FILE and filters alignment with conditional probability greater than TRESHOLD
"""

import sys

file = sys.argv[1]
treshold = float(sys.argv[2])

for line in open(file).readlines():
    alignment, words = line.split('\t')
    for a, w in zip(alignment.split(), words.split()):
        t = w.split('-')[-1]
        if float(t) >= treshold:
            sys.stdout.write(a + ' ')
    print()