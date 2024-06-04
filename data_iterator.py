from math import ceil
import numpy as np
from random import shuffle
import copy
from augmentation import *
import logging


class Data(object):
    def __init__(self, data_name, max_len, logger):
        self.logger = logger

        file = f"./data/{data_name}.txt"

        self.max_len = max_len


        self.pos_seqs, self.num_users, self.num_items = self.read_file(file=file)

        temp_num = self.renumber()
        re_num = self.renumber()
        while re_num != temp_num:
            temp_num = re_num
            re_num = self.renumber()

        self.num_users, self.num_items = len(self.pos_seqs), re_num - len(self.pos_seqs)
        
        inters = sum([len(seq) for seq in self.pos_seqs])
        logger.info("-------------after renumber------------")
        logger.info('users:'+ str(self.num_users))
        logger.info('items:'+ str(self.num_items))
        logger.info('average length:' + str(inters / self.num_users))
        logger.info("data sparsity:" + str(inters / self.num_users / self.num_items))


    def renumber(self, min_freq = 5):
        lens = [len(seq) for seq in self.pos_seqs]
        self.pos_seqs = [seq for seq, seqlen in zip(self.pos_seqs, lens) if seqlen >= min_freq] 
        his_seqs = self.pos_seqs.copy()
        item_cnt = dict()
        for seq in his_seqs:
            for item in seq:
                if item in item_cnt.keys():
                    item_cnt[item] += 1
                else:
                    item_cnt[item] = 1
        item_set = set()
        for item in item_cnt.keys():
            if item_cnt[item] >= min_freq:
                item_set.add(item)

        iid_nbr_dict = {iid: idx for idx, iid in enumerate(sorted(list(item_set)))}
        self.pos_seqs = [[iid_nbr_dict[iid] + 1 for iid in seq if iid in item_set]  for seq in
                         self.pos_seqs]

        return len(item_set) + len(self.pos_seqs)

        
    def read_file(self, file):
        max_item = 0
        max_uid = 0
        pos_seqs = []
        len_list = []
        with open(file, 'r') as f:
            for line in f:
                inter = line.split(' ')
                uid = int(inter[0])
                seq = [int(item) for item in inter[1:]]
                max_item = max(max_item, max(seq))
                max_uid = max(max_uid, uid)
                len_list.append(len(seq))
                pos_seqs.append(seq)
        
        self.logger.info("-------raw data-----")
        self.logger.info('users:' + str(max_uid))
        self.logger.info('items:' + str(max_item))
        self.logger.info('average length:' + str(sum(len_list) / len(len_list)))

        return pos_seqs, max_uid, max_item


class TrainData(Data):
    def __init__(self, config):
        super(TrainData, self).__init__(config.dataset, config.maxlen, config.logger)
        logger = config.logger
        self.pre_seq_aug = config.pre_seq_aug
        self.batch_size = config.batch_size
        self.cl_type = config.aug_type
        self.pos_seqs = [seq[:-2] for seq in self.pos_seqs]
        self.is_hard = config.is_hard

        self.num_users = len(self.pos_seqs)
        
        inters = sum([len(seq) for seq in self.pos_seqs])
        logger.info("-------------for training------------")
        logger.info('users:'+ str(self.num_users))
        logger.info('items:'+ str(self.num_items))
        logger.info('average length:' + str(inters / self.num_users))
        logger.info("data sparsity:" + str(inters / self.num_users / self.num_items))




        self.n_batch = len(self.pos_seqs) // self.batch_size
        self.curr_batch = 0
        if len(self.pos_seqs) % self.batch_size:
            self.n_batch += 1

        if self.is_hard:
            self.base_transform = { 'mask':Mask(config.mask_p), 
                                    'reorder':Reorder(config.reorder_p), 
                                    'crop':Crop(config.crop_p)}
        else:
            self.base_transform = { 'mask':Mask(config.mask_p, is_hard=False), 
                                    'reorder':Reorder(config.reorder_p, is_hard=False), 
                                    'crop':Crop(config.crop_p)}

        self.transform_map = {'m':'mask', 'r':'reorder', 'c':'crop'}



    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_batch >= self.n_batch:
            shuffle(self.pos_seqs)
            self.curr_batch = 0
            raise StopIteration


        idxs = np.arange(self.curr_batch * self.batch_size, 
                        min(len(self.pos_seqs), (self.curr_batch + 1) * self.batch_size))

        in_seqs, out_seqs, out_negs = [], [], []
        for idx in idxs:
            seq = self.pos_seqs[idx].copy()
            
            in_seq = seq[:-1]
            out_seq = seq[1:]
            negs = []

            for _ in in_seq:
                neg = np.random.randint(1, self.num_items + 1)
                while neg in seq:
                    neg = np.random.randint(1, self.num_items + 1)
                negs.append(neg)

            in_seqs.append(in_seq)
            out_seqs.append(out_seq)
            out_negs.append(negs)
            

        lens = [len(seq) for seq in in_seqs]
        max_len = min(self.max_len, max(lens))
        
        lens = [l if l <= max_len else max_len for l in lens]
        
        seqs = [seq + [0] * (max_len - len(seq)) if len(seq) <= max_len else seq[-max_len:] for seq in in_seqs]
        poss = [pos + [0] * (max_len - len(pos)) if len(pos) <= max_len else pos[-max_len:] for pos in out_seqs]
        negs = [neg + [0] * (max_len - len(neg)) if len(neg) <= max_len else neg[-max_len:] for neg in out_negs]


        in_seqs1, in_seqs2 = self.augment(in_seqs)
        aug_seqs1, lens1  = self.seqs_pad(in_seqs1)
        aug_seqs2, lens2 = self.seqs_pad(in_seqs2)
        aug_seqs = [aug_seqs1, aug_seqs2]
        aug_lens = [lens1, lens2]
    
        self.curr_batch += 1

        return seqs, poss, negs, lens, aug_seqs, aug_lens

    def augment(self, in_seqs):
        seqs1, seqs2 = [], []
        if self.cl_type in self.base_transform.keys():
            aug = self.base_transform[self.cl_type]
            for seq in in_seqs:
                seq1, seq2 = aug(seq), aug(seq)
                seqs1.append(seq1)
                seqs2.append(seq2)
        else:
            transform_list = []
            for t in self.cl_type:
                transform_list.append(self.transform_map[t])
            for seq in in_seqs:
                aug_method = np.random.choice(transform_list, size=2, replace=True)
                aug1, aug2 = self.base_transform[aug_method[0]], self.base_transform[aug_method[1]]
                seq1, seq2 = aug1(seq), aug2(seq)
                seqs1.append(seq1)
                seqs2.append(seq2)

        return seqs1, seqs2

    def seqs_pad(self, seqs):
        lens = [len(seq) for seq in seqs]
        max_len = self.max_len

        lens = [l if l <= max_len else max_len for l in lens]
        seqs = [seq + [0] * (max_len - len(seq)) if len(seq) <= max_len else seq[-max_len:] for seq in seqs]
        return seqs, lens



class TestData(Data):
    def __init__(self, config, is_valid):
        super(TestData, self).__init__(config.dataset, config.maxlen, config.logger)
        self.batch_size = config.batch_size
        self.n_batch = len(self.pos_seqs) // config.batch_size
        if len(self.pos_seqs) % self.batch_size:
            self.n_batch += 1
        
        if is_valid:
            self.pos_seqs = [seq[:-1] for seq in self.pos_seqs]
        self.curr_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_batch >= self.n_batch:
            self.curr_batch = 0
            raise StopIteration

        idxs = np.arange(self.curr_batch * self.batch_size, 
                        min(len(self.pos_seqs), (self.curr_batch + 1) * self.batch_size))

        seqs, tars = [], []
        for idx in idxs:
            seq = self.pos_seqs[idx]
            seq_in = seq[:-1]
            seqs.append(seq_in)
            tars.append(seq[-1])

        max_len = min(self.max_len, max(list(map(len, seqs))))
        lens = [len(seq) if len(seq) <= max_len else max_len for seq in seqs]
        seqs = [seq + [0] * (max_len - len(seq)) if len(seq) <= self.max_len else seq[-max_len:] for seq in seqs]
        self.curr_batch += 1

        return seqs, tars, lens
    
