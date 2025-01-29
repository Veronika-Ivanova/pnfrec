"""
Pytorch Lightning Modules.
"""
import sys
sys.path.append('/home/jovyan/ivanova/negative_feedback/src')

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from models import SASRecContrastiveBaseline


class InfoNCE(nn.Module):
    """
    The following implementation is adapted from: https://github.com/RElbers/info-nce-pytorch

    
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', ignore_index = 0):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.ignore_index = ignore_index

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        ignore_index=self.ignore_index)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired',
             ignore_index=0):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class SeqRecBase(pl.LightningModule):

    def __init__(self, model, lr, padding_idx=0,
                 predict_top_k=10, filter_seen=True, neg_CE_coef=1, contrastive_coef=1):

        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.filter_seen = filter_seen
        self.neg_CE_coef = neg_CE_coef
        self.contrastive_coef = contrastive_coef

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)

        scores = scores.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        user_ids = batch['user_id'].detach().cpu().numpy()

        return {'preds': preds, 'scores': scores, 'user_ids': user_ids}

    def validation_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)
        metrics = self.compute_val_metrics(batch['target'], preds)

        self.log("val_ndcg", metrics['ndcg'], prog_bar=True)
        self.log("val_hit_rate", metrics['hit_rate'], prog_bar=True)
        self.log("val_mrr", metrics['mrr'], prog_bar=True)

    def make_prediction(self, batch):

        outputs = self.prediction_output(batch)

        input_ids = batch['input_ids']
        rows_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1

        preds = outputs[rows_ids, last_item_idx, :]

        scores, preds = torch.sort(preds, descending=True)

        if self.filter_seen:
            seen_items = batch['full_history']
            preds, scores = self.filter_seen_items(preds, scores, seen_items)
        else:
            scores = scores[:, :self.predict_top_k]
            preds = preds[:, :self.predict_top_k]

        return preds, scores

    def filter_seen_items(self, preds, scores, seen_items):

        max_len = seen_items.size(1)
        scores = scores[:, :self.predict_top_k + max_len]
        preds = preds[:, :self.predict_top_k + max_len]

        final_preds, final_scores = [], []
        for i in range(preds.size(0)):
            not_seen_indexes = torch.isin(preds[i], seen_items[i], invert=True)
            pred = preds[i, not_seen_indexes][:self.predict_top_k]
            score = scores[i, not_seen_indexes][:self.predict_top_k]
            final_preds.append(pred)
            final_scores.append(score)

        final_preds = torch.vstack(final_preds)
        final_scores = torch.vstack(final_scores)

        return final_preds, final_scores

    def compute_val_metrics(self, targets, preds):

        ndcg, hit_rate, mrr = 0, 0, 0

        for i, pred in enumerate(preds):
            if torch.isin(targets[i], pred).item():
                hit_rate += 1
                rank = torch.where(pred == targets[i])[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
                mrr += 1 / rank

        hit_rate = hit_rate / len(targets)
        ndcg = ndcg / len(targets)
        mrr = mrr / len(targets)

        return {'ndcg': ndcg, 'hit_rate': hit_rate, 'mrr': mrr}


class SeqRec(SeqRecBase):

    def training_step(self, batch, batch_idx):

        model_outputs = self.model(batch)
        outputs = model_outputs['outputs']
        loss = self.compute_CE_loss(outputs, batch['labels'])

        return loss
    
    def compute_CE_loss(self, outputs, labels):

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        return loss

    def prediction_output(self, batch):

        return self.model(batch)

    
class SeqRecContrastiveBaseline(SeqRec):

    def training_step(self, batch, batch_idx):
        model_outputs = self.model(batch)
        outputs = model_outputs['outputs']
        CE_loss = self.compute_CE_loss(outputs, batch['labels'])
        
        x_ = model_outputs['seqs']
        skip = batch['skip']
        input_ids = batch['input_ids']
        
        neg_targets, pos_query, pos_key = SASRecContrastiveBaseline.get_negative_samples(self.model, x_, skip, input_ids)
        skip_loss = self.compute_skip_loss(neg_targets, pos_query, pos_key)
        
        loss = CE_loss + self.contrastive_coef * skip_loss

        return loss
    
    def compute_skip_loss(self, neg_targets, pos_query, pos_key):
        
        skip_loss = InfoNCE(negative_mode='paired', ignore_index=self.padding_idx)   
        loss = skip_loss(pos_query, pos_key, negative_keys=neg_targets)
        
        return loss

    
class SeqRecPosNeg(SeqRec):
    
    def training_step(self, batch, batch_idx):
        
        model_outputs = self.model(batch)
        positive_outputs = model_outputs['positive_outputs']
        positive_CE = self.compute_CE_loss(positive_outputs, batch['labels'])
        
        negative_outputs = model_outputs['negative_outputs']
        negative_CE = self.compute_CE_loss(negative_outputs, batch['negative_labels'])
        
        loss = positive_CE + self.neg_CE_coef * negative_CE

        return loss

    
class SeqRecPosContrastive(SeqRec):
    
    def training_step(self, batch, batch_idx):
        
        model_outputs = self.model(batch)
        positive_outputs = model_outputs['positive_outputs']
        positive_CE = self.compute_CE_loss(positive_outputs, batch['labels'])
        
        next_positive_sim = model_outputs['next_positive_sim'] 
        masked_positive_contrastive_sim = model_outputs['masked_positive_contrastive_sim']
        positive_contrastive = self.compute_contrastive_loss(next_positive_sim, masked_positive_contrastive_sim)
        
        loss = positive_CE + self.contrastive_coef * positive_contrastive

        return loss
    
    def compute_contrastive_loss(self, next_item_sim, contrastive_sim):
        
        loss = ((contrastive_sim.sum(axis=1) + next_item_sim.exp()).log() - next_item_sim).mean()
        
        return loss


class SeqRecPosNegContrastive(SeqRecPosContrastive):
    
    def training_step(self, batch, batch_idx):
        
        model_outputs = self.model(batch)
        positive_outputs = model_outputs['positive_outputs']
        positive_CE = self.compute_CE_loss(positive_outputs, batch['labels'])
        
        negative_outputs = model_outputs['negative_outputs']
        negative_CE = self.compute_CE_loss(negative_outputs, batch['negative_labels'])
        
        next_positive_sim = model_outputs['next_positive_sim'] 
        masked_positive_contrastive_sim = model_outputs['masked_positive_contrastive_sim']
        positive_contrastive = self.compute_contrastive_loss(next_positive_sim, masked_positive_contrastive_sim)
        
        loss = positive_CE + self.neg_CE_coef * negative_CE + self.contrastive_coef * positive_contrastive

        return loss
