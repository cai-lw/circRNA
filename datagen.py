from random import sample
from itertools import cycle, islice, tee
from collections import deque
import numpy as np

MAXLEN = 100000

np.random.seed(233)

# Consume x_gen first. y's will be cached while consuming.
class xy_gen(object):
    def __init__(self):
        self.y_buf = []

    def x_gen(self, *args, **kwargs):
        def g():
            dg = data_gen(*args, **kwargs)
            while True:
                try:
                    x, y = next(dg)
                except StopIteration:
                    return
                self.y_buf.append(y)
                yield x
        return g()

    def y_array(self, n=None, axis=0):
        ret = np.concatenate(self.y_buf, axis)
        if n is None:
            return ret
        else:
            return np.split(ret, [n], axis)[0]

def data_gen(subgroups, sample_per_iter, use_alu):
    ret = []
    label = []
    for i_group in cycle(subgroups):
        with open('clean_data/pos%d.txt' % i_group) as f_pos, open('clean_data/neg%d.txt' % i_group) as f_neg:
            pos_seqs = list(map(lambda s:s.strip(), f_pos.readlines()))
            neg_seqs = list(map(lambda s:s.strip(), f_neg.readlines()))
        n_total = len(pos_seqs) + len(neg_seqs)
        n_pos = len(pos_seqs)
        for i in np.random.permutation(n_total):
            if i < n_pos:
                ret.append(pos_seqs[i])
                label.append(1)
            else:
                ret.append(neg_seqs[i - n_pos])
                label.append(0)
            if len(ret) >= sample_per_iter:
                yield (to_x(ret, use_alu), np.array(label))
                ret.clear()
                label.clear()

dna_dict = {'A' : 0, 'a': 0,
            'T' : 1, 't': 1,
            'C' : 2, 'c': 2,
            'G' : 3, 'g': 3}

def to_x(seqs, use_alu):
    n_samples = len(seqs)
    feature_dim = 5 if use_alu else 4
    enum_lim = lambda seq: enumerate(islice(seq, MAXLEN))
    index_seqs = [list(map(lambda t:t[0] * feature_dim + dna_dict[t[1]], enum_lim(seq))) for seq in seqs]
    data = np.zeros((n_samples, MAXLEN, feature_dim), dtype=np.float32)
    for i, seq in enumerate(index_seqs):
        np.put(data[i, :, :], seq, 1.0)
    if use_alu:
        alu_index_seqs = [list(map(lambda t:t[0], filter(lambda t:t[1].islower(), enum_lim(seq)))) for seq in seqs]
        for i, seq in enumerate(alu_index_seqs):
            np.put(data[i, :, 4], seq, 1.0)
    return data
