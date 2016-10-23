from random import sample
from itertools import cycle, islice, tee
from collections import deque
import numpy as np

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
                    x, y, w = next(dg)
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

def data_gen(subgroups, sample_per_iter, maxlen, use_alu, cut=False, sample_per_group=None, pos_ratio=None):
    ret = []
    label = []
    weight = []
    for i_group in cycle(subgroups):
        with open('clean_data/pos%d.txt' % i_group) as f_pos, open('clean_data/neg%d.txt' % i_group) as f_neg:
            if cut:
                rs = np.random.RandomState(1234567 + i_group)
                def random_cut(s):
                    if len(s) <= maxlen:
                        return s
                    else:
                        ix = rs.randint(len(s) - maxlen)
                        return s[ix:(ix + maxlen)]
                pos_seqs = [s.strip()[:]]
                pos_seqs = [random_cut(s.strip()) for s in f_pos.readlines()]
                neg_seqs = [random_cut(s.strip()) for s in f_neg.readlines()]
            else:
                pos_seqs = [s.strip() for s in f_pos.readlines() if len(s) <= maxlen]
                neg_seqs = [s.strip() for s in f_neg.readlines() if len(s) <= maxlen]
        n_pos = len(pos_seqs)
        n_neg = len(neg_seqs)
        if sample_per_group is None:
            sample_per_group = n_pos + n_neg
        rs = np.random.RandomState(2333333 + i_group)
        if pos_ratio is None:
            w_pos = n_neg / n_pos
            idx = rs.permutation(n_pos + n_neg)[:sample_per_group]
        else:
            w_pos = (1 - pos_ratio) / pos_ratio
            n_smpl_pos = min(int(pos_ratio * sample_per_group), n_pos, int(n_neg / (1 - pos_ratio) * pos_ratio))
            n_smpl_neg = min(int((1 - pos_ratio) * sample_per_group), n_neg, int(n_pos / pos_ratio * (1 - pos_ratio)))
            idx_pos = rs.permutation(n_pos)[:n_smpl_pos]
            idx_neg = rs.permutation(n_neg)[:n_smpl_neg] + n_pos
            idx = rs.permutation(np.concatenate((idx_pos, idx_neg)))
        for i in idx:
            if i < n_pos:
                ret.append(pos_seqs[i])
                label.append(1)
                weight.append(w_pos)
            else:
                ret.append(neg_seqs[i - n_pos])
                label.append(0)
                weight.append(1)
            if len(ret) >= sample_per_iter:
                yield (to_x(ret, use_alu), np.array(label), np.array(weight))
                del ret[:]
                del label[:]
                del weight[:]

dna_dict = {'A' : 0, 'a': 0,
            'T' : 1, 't': 1,
            'C' : 2, 'c': 2,
            'G' : 3, 'g': 3}

def to_x(seqs, use_alu):
    n_samples = len(seqs)
    feature_dim = 5 if use_alu else 4
    index_seqs = [[t[0] * feature_dim + dna_dict[t[1]] for t in enumerate(seq)] for seq in seqs]
    data = np.zeros((n_samples, max(len(s) for s in seqs), feature_dim), dtype=np.float32)
    for i, seq in enumerate(index_seqs):
        np.put(data[i, :, :], seq, 1.0)
    if use_alu:
        alu_index_seqs = [[t[0] for t in filter(lambda t:t[1].islower(), enumerate(seq))] for seq in seqs]
        for i, seq in enumerate(alu_index_seqs):
            np.put(data[i, :, 4], seq, 1.0)
    return data
