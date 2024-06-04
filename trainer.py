from re import L
import torch
import os
import numpy as np

from tqdm import tqdm

from tool import *

class Trainer(object):
    def __init__(self) -> None:
        super().__init__()

    def train(self, epoch, writer, model, train_data, config):
        logger = config.logger
        model.train()

        step = 0
        loss_sum = 0

        if not os.path.exists('runs'):
            os.mkdir('runs')

        for seqs, poss, negs, lens, aug_seqs, aug_lens in train_data:
            # torch.cuda.empty_cache()
            seqs = trans_to_cuda(torch.LongTensor(seqs))
            poss = trans_to_cuda(torch.LongTensor(poss))
            negs = trans_to_cuda(torch.LongTensor(negs))
            lens = trans_to_cuda(torch.LongTensor(lens))

            model.optimizer.zero_grad()

            loss = 0
            cl_loss_record = 0
            step += 1


            # calculate the Next item loss           
            loss += model.cal_loss(seqs, poss, negs, lens)
                    
            # calculate the contrastive loss
            aug_seqs1, aug_seqs2 = trans_to_cuda(torch.LongTensor(aug_seqs[0])), trans_to_cuda(torch.LongTensor(aug_seqs[1]))
            aug_lens1, aug_lens2 = trans_to_cuda(torch.LongTensor(aug_lens[0])), trans_to_cuda(torch.LongTensor(aug_lens[1]))
            
            aug_embs1 = model(aug_seqs1, aug_lens1, phase=config.cl_embs)
            aug_embs2 = model(aug_seqs2, aug_lens2, phase=config.cl_embs)
            batch_size = aug_embs1.shape[0]

            if config.cl_embs == 'mean':
                cl_loss_record += model.cl_loss(aug_embs1.mean(-2), aug_embs2.mean(-2)) * config.w_clloss
            elif config.cl_embs == 'concat':
                cl_loss_record += model.cl_loss(aug_embs1.view(batch_size, -1), aug_embs2.view(batch_size, -1)) * config.w_clloss
            else:
                cl_loss_record += model.cl_loss(aug_embs1, aug_embs2) * config.w_clloss
            
            loss += cl_loss_record
            loss.backward()
            model.optimizer.step()
            loss_sum += loss.item()
            writer.add_scalar("loss", loss.item(), step)

        logger.info('Epoch(by epoch):{:d}\tloss:{:4f}'\
            .format(epoch, loss_sum / train_data.n_batch / config.test_epoch))


    def eval(self, epoch, model, config, test_data, ks, phase='valid'):
        logger = config.logger

        recall, ndcg = [0] * len(ks), [0] * len(ks)
        num = 0
        model.eval()

        test_data_iter = tqdm(test_data, total=test_data.n_batch)
        with torch.no_grad():
            for seqs, tars, lens in test_data_iter:
                seqs = trans_to_cuda(torch.LongTensor(seqs))
                lens = trans_to_cuda(torch.LongTensor(lens))
                item_scores = model.full_sort_predict(seqs, lens)
                nrecall = int(ks[-1])
                item_scores[:, 0] -= 1e9
                if config.repeat_rec == False:
                    for seq, item_score in zip(seqs, item_scores):
                        item_score[seq] -= 1e9
                _, items = torch.topk(item_scores, nrecall, sorted=True)
                items = trans_to_cpu(items).detach().numpy()

                batch_recall, batch_ndcg = [0] * len(ks), [0] * len(ks)

                for item, tar in zip(items, tars):
                    for k, kk in enumerate(ks):
                        if tar in set(item[:kk]):
                            batch_recall[k] += 1
                            item_idx = {i:idx + 1 for idx, i in enumerate(item[:kk])}
                            batch_ndcg[k] += (1 / np.log2(item_idx[tar] + 1))
                
                recall = [r + br for r, br in zip(recall, batch_recall)]
                ndcg = [n + bn for n, bn in zip(ndcg, batch_ndcg)]
                num += seqs.shape[0] 
                
        if phase == 'valid':
            log_str = 'Valid: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'Recall@{:2d}:\t{:.4f}\t'.format(kk, recall[nbr_k] / num)
            logger.info(log_str)
            log_str = 'Valid: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'NDCG@{:2d}:\t{:.4f}\t'.format(kk, ndcg[nbr_k] / num)
            logger.info(log_str)
        else:
            log_str = 'Test: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'Recall@{:2d}:\t{:.4f}\t'.format(kk, recall[nbr_k] / num)
            logger.info(log_str)
            log_str = 'Test: '
            for nbr_k, kk in enumerate(ks):
                log_str += 'NDCG@{:2d}:\t{:.4f}\t'.format(kk, ndcg[nbr_k] / num)
            logger.info(log_str)

        if ks is None:
            return [recall / num]
        else:
            return [r / num for r in recall]
