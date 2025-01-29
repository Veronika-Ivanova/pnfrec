"""
Models.
"""

import numpy as np
import torch
from torch import nn


class SASRec(nn.Module):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads=1, initializer_range=0.02,
                 add_head=True):

        super(SASRec, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(hidden_units,
                                                   num_heads,
                                                   dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def _apply_sequential_encoder(self, input_ids, attention_layernorms, attention_layers, forward_layernorms, forward_layers):
        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        # need to be on the same device
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.Tensor(input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        # need to be on the same device
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))

        for i in range(self.num_blocks):
            seqs = torch.transpose(seqs, 0, 1)
            Q = attention_layernorms[i](seqs)

            mha_outputs, _ = attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = forward_layernorms[i](seqs)
            seqs = forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        return seqs

    def forward(self, batch):
        input_ids = batch['input_ids']
        seqs = self._apply_sequential_encoder(input_ids, 
                                  self.attention_layernorms, 
                                  self.attention_layers, 
                                  self.forward_layernorms, 
                                  self.forward_layers)
        seqs = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            outputs = torch.matmul(seqs, self.item_emb.weight.transpose(0, 1))
        
        if self.training:            
            return {
                'outputs': outputs,
                'seqs': seqs,
            }
        else:
            return outputs
    
    
class SASRecContrastiveBaseline(SASRec):
    def __init__(self, item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads=1, initializer_range=0.02,
                 add_head=True):

        super().__init__(item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads, initializer_range,
                 add_head)
            
    def get_negative_samples(self, a, skip, input_ids=None):
        if input_ids is None:
            keys = a
        else:
            keys = self.item_emb(input_ids)
        neg_mask = skip == 1
        pos_mask = skip == 0

        n = torch.zeros_like(a)
        n[neg_mask] = keys[neg_mask]
        
        p = torch.zeros_like(a)
        p_key = torch.zeros_like(a)
        
        #ignore last item which does not have a closest positive sample
        p_key[:, :-1][pos_mask[:, :-1]] = keys[torch.arange(a.shape[0]).unsqueeze(-1),
                                    self._closest_pos_sample(a, skip)][:, :-1][pos_mask[:, :-1]] # на месте айтема с позитивным фидбеком его выход энкодера
        p[:, :-1][pos_mask[:, :-1]] = a[:, :-1][pos_mask[:, :-1]] # на месте айтема с позитивным фидбеком его эмбед
        # n на месте айтема с негативным фидбеком его эмбед
        n_size = self.maxlen - 1
                                                                        
        return n.tile(n.shape[1], 1, 1), p.view(-1, *p.shape[2:]), p_key.view(-1, *p_key.shape[2:])
    
    def _closest_pos_sample(self, a, skip):
        m = torch.iinfo(skip.dtype).max

        z = torch.linspace(0, a.shape[1] - 1, a.shape[1]).unsqueeze(dim=-1)
        z = z.tile((a.shape[0],)).transpose(0, 1)

        b = z.detach().clone().unsqueeze(dim=-1)

        z[skip != 0] = m

        z = z.unsqueeze(dim=-1).tile((1, 1, a.shape[1])).transpose(1, 2)

        diff = (z - b)

        diff[diff <= 0] = m

        return diff.argmin(dim=2)
    
    
class SASRecPosNeg(SASRec):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads=1, initializer_range=0.02,
                 add_head=True):

        super().__init__(item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads, initializer_range,
                 add_head)

        self.positive_attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.positive_attention_layers = nn.ModuleList()
        self.positive_forward_layernorms = nn.ModuleList()
        self.positive_forward_layers = nn.ModuleList()
        
        self.negative_attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.negative_attention_layers = nn.ModuleList()
        self.negative_forward_layernorms = nn.ModuleList()
        self.negative_forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            positive_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.positive_attention_layernorms.append(positive_attn_layernorm)

            positive_attn_layer = nn.MultiheadAttention(hidden_units,
                                                        num_heads,
                                                        dropout_rate)
            self.positive_attention_layers.append(positive_attn_layer)

            positive_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.positive_forward_layernorms.append(positive_fwd_layernorm)

            positive_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.positive_forward_layers.append(positive_fwd_layer)
            
        for _ in range(num_blocks):
            negative_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.negative_attention_layernorms.append(negative_attn_layernorm)

            negative_attn_layer = nn.MultiheadAttention(hidden_units,
                                                        num_heads,
                                                        dropout_rate)
            self.negative_attention_layers.append(negative_attn_layer)

            negative_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.negative_forward_layernorms.append(negative_fwd_layernorm)

            negative_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.negative_forward_layers.append(negative_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)    

    def forward(self, batch):
        ###### positives
        positive_input_ids = batch['input_ids']
        positive_seqs = self._apply_sequential_encoder(positive_input_ids, 
                                  self.positive_attention_layernorms, 
                                  self.positive_attention_layers, 
                                  self.positive_forward_layernorms, 
                                  self.positive_forward_layers)
        positive_seqs = self.last_layernorm(positive_seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            positive_outputs = torch.matmul(positive_seqs, self.item_emb.weight.transpose(0, 1))
            
        if self.training:
            ###### negatives
            negative_input_ids = batch['negative_input_ids']
            negative_seqs = self._apply_sequential_encoder(negative_input_ids, 
                                      self.negative_attention_layernorms, 
                                      self.negative_attention_layers, 
                                      self.negative_forward_layernorms, 
                                      self.negative_forward_layers)

            negative_seqs = self.last_layernorm(negative_seqs) # (U, T, C) -> (U, -1, C)
            if self.add_head:
                negative_outputs = torch.matmul(negative_seqs, self.item_emb.weight.transpose(0, 1))
                
            return {
                    'positive_outputs': positive_outputs,
                    'negative_outputs': negative_outputs
            }
        else:
            return positive_outputs
        

class SASRecPosContrastive(SASRec):
    def __init__(self, item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads=1, initializer_range=0.02,
                 add_head=True):

        super().__init__(item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads, initializer_range,
                 add_head)
        
    def _get_contrastive_loss_inputs(self, input_ids, contrastive_input_ids, labels, seqs):
        mask = input_ids != 0
        contrastive_mask = contrastive_input_ids != 0

        labels_emb = self.item_emb(labels[mask])
        next_item_sim = torch.einsum(
            'md,md->m',
            seqs[mask].reshape(-1, self.hidden_units),
            labels_emb
        )

        contrastive_items_emb = self.item_emb(contrastive_input_ids)
        contrastive_sim = torch.einsum(
            'mnd,mkd->mnk',
            seqs,
            contrastive_items_emb
        )

        pad_mask = (contrastive_mask
                    .reshape(contrastive_input_ids.shape[0], contrastive_input_ids.shape[1], 1)
                    .tile(input_ids.shape[1])
                    .mT[mask])

        masked_contrastive_sim = torch.einsum(
            'mn,mn->mn',
            contrastive_sim[mask].exp(),
            pad_mask
        )
        return next_item_sim, masked_contrastive_sim
    
    def forward(self, batch):
        ###### positives
        positive_input_ids = batch['input_ids']
        positive_seqs = self._apply_sequential_encoder(positive_input_ids, 
                                  self.attention_layernorms, 
                                  self.attention_layers, 
                                  self.forward_layernorms, 
                                  self.forward_layers)
        positive_seqs = self.last_layernorm(positive_seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            positive_outputs = torch.matmul(positive_seqs, self.item_emb.weight.transpose(0, 1))
            
        if self.training:
            ###### negatives
            negative_input_ids = batch['negative_input_ids']
            
            ##### positive contrastive
            positive_labels = batch['labels']  
            negative_labels = batch['negative_labels']
            
            next_positive_sim, masked_positive_contrastive_sim = self._get_contrastive_loss_inputs(
                positive_input_ids, negative_input_ids, positive_labels, positive_seqs
            )
            
            return {
                    'positive_outputs': positive_outputs,
                    'next_positive_sim': next_positive_sim,
                    'masked_positive_contrastive_sim': masked_positive_contrastive_sim
            }
        else:
            return positive_outputs
    
        
class SASRecPosNegContrastive(SASRecPosContrastive):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads=1, initializer_range=0.02,
                 add_head=True):

        super().__init__(item_num, maxlen, hidden_units, num_blocks,
                 dropout_rate, num_heads, initializer_range,
                 add_head)

        self.positive_attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.positive_attention_layers = nn.ModuleList()
        self.positive_forward_layernorms = nn.ModuleList()
        self.positive_forward_layers = nn.ModuleList()
        
        self.negative_attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.negative_attention_layers = nn.ModuleList()
        self.negative_forward_layernorms = nn.ModuleList()
        self.negative_forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            positive_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.positive_attention_layernorms.append(positive_attn_layernorm)

            positive_attn_layer = nn.MultiheadAttention(hidden_units,
                                                        num_heads,
                                                        dropout_rate)
            self.positive_attention_layers.append(positive_attn_layer)

            positive_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.positive_forward_layernorms.append(positive_fwd_layernorm)

            positive_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.positive_forward_layers.append(positive_fwd_layer)
            
        for _ in range(num_blocks):
            negative_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.negative_attention_layernorms.append(negative_attn_layernorm)

            negative_attn_layer = nn.MultiheadAttention(hidden_units,
                                                        num_heads,
                                                        dropout_rate)
            self.negative_attention_layers.append(negative_attn_layer)

            negative_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.negative_forward_layernorms.append(negative_fwd_layernorm)

            negative_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.negative_forward_layers.append(negative_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)    

    def forward(self, batch):
        ###### positives
        positive_input_ids = batch['input_ids']
        positive_seqs = self._apply_sequential_encoder(positive_input_ids, 
                                  self.positive_attention_layernorms, 
                                  self.positive_attention_layers, 
                                  self.positive_forward_layernorms, 
                                  self.positive_forward_layers)
        positive_seqs = self.last_layernorm(positive_seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            positive_outputs = torch.matmul(positive_seqs, self.item_emb.weight.transpose(0, 1))
            
        if self.training:
            ###### negatives
            negative_input_ids = batch['negative_input_ids']
            negative_seqs = self._apply_sequential_encoder(negative_input_ids, 
                                      self.negative_attention_layernorms, 
                                      self.negative_attention_layers, 
                                      self.negative_forward_layernorms, 
                                      self.negative_forward_layers)

            negative_seqs = self.last_layernorm(negative_seqs) # (U, T, C) -> (U, -1, C)
            if self.add_head:
                negative_outputs = torch.matmul(negative_seqs, self.item_emb.weight.transpose(0, 1))
            
            ##### positive contrastive
            positive_labels = batch['labels']  
            negative_labels = batch['negative_labels']
            
            next_positive_sim, masked_positive_contrastive_sim = self._get_contrastive_loss_inputs(
                positive_input_ids, negative_input_ids, positive_labels, positive_seqs
            )
            
            return {
                    'positive_outputs': positive_outputs,
                    'negative_outputs': negative_outputs,
                    'next_positive_sim': next_positive_sim,
                    'masked_positive_contrastive_sim': masked_positive_contrastive_sim
            }
        else:
            return positive_outputs


class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
