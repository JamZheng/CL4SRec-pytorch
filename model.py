from numpy import zeros
import torch
from torch import nn
from layers import TransformerEncoder
from utils import *

class CL4SRec(nn.Module):
    def __init__(self, config):
        super(CL4SRec, self).__init__()

        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size  # same as embedding_size
        self.inner_size = config.inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attn_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = 1e-12
        self.batch_size = config.batch_size

        self.initializer_range = 0.02

        self.loss_type = config.loss_type
        self.n_item = config.n_item
        self.max_seq_length = config.maxlen
        self.temp = config.temp

        # define layers and loss
        self.item_embeddings = nn.Embedding(self.n_item, self.hidden_size, padding_idx=0)
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.cl_loss = ContrastiveLoss(config)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # 参数的正则项系数
        # parameters initialization

        self.apply(self._init_weights)
        if self.loss_type == 'BCE':
            self.criterion = nn.BCELoss(reduction='mean')
        elif self.loss_type == 'MCE' or self.loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE' ...]!")

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        # [B, seq_len]
        attention_mask = (item_seq > 0).long()
        
        # [B, 1, 1, seq_len]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len, phase='predict'):
        # position embedding 
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)        
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        # get positon embedding 
        item_emb = self.item_embeddings(item_seq)
        input_emb = item_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        if phase == 'predict':    
            output = self.gather_indexes(output, item_seq_len - 1)

        return output  # [B H]

    def cross_entropy(self, seq_out, poss, negs):
        # [batch seq_len hidden_size]
        seq_len = poss.shape[1]
        pos_emb = self.item_embeddings(poss)
        neg_emb = self.item_embeddings(negs)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1) # [batch*seq_len]
        istarget = (poss > 0).view(poss.size(0) * seq_len).float()  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def cal_loss(self, input_ids, poss, negs, lens):
        if self.loss_type == 'BCE':
            seq_embs = self.forward(input_ids, lens, phase='concat')
            loss = self.cross_entropy(seq_embs, poss, negs)
        return loss


    def full_sort_predict(self, item_seq, item_seq_len):
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embeddings.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)