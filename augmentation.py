from math import ceil, floor
from typing import Any
import numpy as np
import copy

class Augment(object):
    def __init__(self, p):
        self.p = p

class Mask(Augment):
    def __init__(self, p, is_hard=True):
        super(Mask, self).__init__(p)
        self.is_hard = is_hard
    
    def __call__(self, ori_seq):
        if self.is_hard:
            return self.hard_mask(ori_seq)
        else:
            return self.soft_mask(ori_seq)

    def soft_mask(self, ori_seq):
        seq = copy.deepcopy(np.array(ori_seq))
        mask = ((np.random.rand(seq.size)) > self.p)
        if mask.sum() < 1:
            mask = ((np.random.rand(seq.size)) > self.p)
        seq[mask] = 0
        return seq.tolist()

    def hard_mask(self, ori_seq):
        seq = copy.deepcopy(np.array(ori_seq))
        seq_idx = np.random.choice(np.arange(0, len(seq)), size=floor(len(seq) * self.p), replace=False)
        # seq_idx = np.sort(seq_idx)
        seq[seq_idx] = 0
        return seq.tolist()

class Reorder(Augment):
    def __init__(self, p, is_hard=True):
        super(Reorder, self).__init__(p)
        self.is_hard = is_hard
    
    def __call__(self, ori_seq):
        if self.is_hard:
            return self.hard_reorder(ori_seq)
        else:
            return self.soft_reorder(ori_seq)

    def hard_reorder(self, ori_seq):
        seq = copy.deepcopy(np.array(ori_seq))
        begin = np.random.randint(0, ceil(len(seq) - (len(seq) * self.p)))
        ori_idx = np.arange(begin, ceil(begin + len(seq) * self.p))
        shuffle_idx = copy.deepcopy(ori_idx)
        np.random.shuffle(shuffle_idx)
        seq[ori_idx] = seq[shuffle_idx]
        return seq.tolist()
    
    def soft_reorder(self, ori_seq):
        seq = copy.deepcopy(np.array(ori_seq))
        ori_idx = np.random.choice(np.arange(0, len(seq)), size=ceil(len(seq) * self.p), replace=False)
        shuffle_idx = copy.deepcopy(ori_idx)
        np.random.shuffle(shuffle_idx)
        seq[ori_idx] = seq[shuffle_idx]
        return seq.tolist()

class Crop(Augment):
    def __init__(self, p):
        super(Crop, self).__init__(p)
    
    def __call__(self, ori_seq):

        seq = copy.deepcopy(np.array(ori_seq))
        begin = np.random.randint(0, ceil(len(seq) - (len(seq) * self.p)))
        tar_idx = np.arange(begin, ceil(begin + len(seq) * self.p))
        seq = seq[tar_idx]


        return seq.tolist()
