from random import sample
from itertools import cycle, islice, tee
from collections import deque
import numpy as np

MAXLEN = 100000

# Consume x_gen first. y's will be cached while consuming.
class xy_gen(object):
    def __init__(self):
        self.y_buf = []
        self.w_buf = []

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

def data_gen(subgroups, sample_per_iter, sample_per_group, use_alu):
    ret = []
    label = []
    for i_group in cycle(subgroups):
        with open('clean_data/pos%d.txt' % i_group) as f_pos, open('clean_data/neg%d.txt' % i_group) as f_neg:
            pos_seqs = [s.strip() for s in f_pos.readlines()]
            neg_seqs = [s.strip() for s in f_neg.readlines()]
        n_pos = len(pos_seqs)
        n_neg = len(neg_seqs)
        rs = np.random.RandomState(2333333 + i_group)
        idx_pos = rs.permutation(n_pos)[:(sample_per_group // 2)]
        idx_neg = rs.permutation(n_neg)[:(sample_per_group // 2)] + n_pos
        for i in rs.permutation(np.concatenate([idx_pos, idx_neg])):
            if i < n_pos:
                ret.append(pos_seqs[i])
                label.append(1)
            else:
                ret.append(neg_seqs[i - n_pos])
                label.append(0)
            if len(ret) >= sample_per_iter:
                yield (to_x(ret, use_alu), np.array(label))
                del ret[:]
                del label[:]

dna_dict = {'A' : 0, 'a': 0,
            'T' : 1, 't': 1,
            'C' : 2, 'c': 2,
            'G' : 3, 'g': 3}

def to_x(seqs, use_alu):
    n_samples = len(seqs)
    feature_dim = 5 if use_alu else 4
    enum_lim = lambda seq: enumerate(islice(seq, MAXLEN))
    index_seqs = [[t[0] * feature_dim + dna_dict[t[1]] for t in enum_lim(seq)] for seq in seqs]
    data = np.zeros((n_samples, MAXLEN, feature_dim), dtype=np.float32)
    for i, seq in enumerate(index_seqs):
        np.put(data[i, :, :], seq, 1.0)
    if use_alu:
        alu_index_seqs = [[t[0] for t in filter(lambda t:t[1].islower(), enum_lim(seq))] for seq in seqs]
        for i, seq in enumerate(alu_index_seqs):
            np.put(data[i, :, 4], seq, 1.0)
    return data
