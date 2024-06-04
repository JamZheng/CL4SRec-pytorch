import torch
import torch.nn as nn
from tool import *

class ContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.temp = config.temp
        self.batch_size = config.batch_size
        self.maxlen = config.maxlen
        self.eps = 1e-12

    def forward(self, pos_embeds1, pos_embeds2):
        batch_size = pos_embeds1.shape[0]
        pos_diff = torch.sum(pos_embeds1 * pos_embeds2, dim=-1).view(-1, 1) / self.temp
        score11 = torch.matmul(pos_embeds1, pos_embeds1.transpose(0, 1)) / self.temp
        score22 = torch.matmul(pos_embeds2, pos_embeds2.transpose(0, 1)) / self.temp
        score12 = torch.matmul(pos_embeds1, pos_embeds2.transpose(0, 1)) / self.temp

        mask = (-torch.eye(batch_size).long() + 1).bool()
        mask = trans_to_cuda(mask)
        score11 = score11[mask].view(batch_size, -1)
        score22 = score22[mask].view(batch_size, -1)
        score12 = score12[mask].view(batch_size, -1)


        score1 = torch.cat((pos_diff, score11, score12), dim=1) # [B, 2B - 2]
        score2 = torch.cat((pos_diff, score22, score12), dim=1)
        score = torch.cat((score1, score2), dim=0)

        labels = torch.zeros(batch_size * 2).long()
        score = trans_to_cuda(score)
        labels = trans_to_cuda(labels)
        assert score.shape[-1] == 2 * batch_size - 1
        return self.ce_loss(score, labels)
    
