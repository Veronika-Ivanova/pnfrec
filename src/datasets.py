"""
Torch datasets and collate function.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CausalDataset(Dataset):
    """Training dataset for negative feedback."""

    padding_value = 0
    labels_padding_value = -100

    def __init__(self, df, max_length,
                 relevance_col, relevance_threshold,
                 user_col, item_col='item_id',
                 time_col='time_idx'):
        """
        Args:
            df (pd.DataFrame): Dataframe with interactions.
            max_length (int): Maximum sequence legth.
            user_col (str): User column in dataframe.
            item_col (str): Item column in dataframe.
            time_col (str): Time column in dataframe.
            relevance_col (str): Relevance column in dataframe.
            relevance_threshold (float): Threshold for negative feedback.
        """

        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.relevance_col = relevance_col
        self.relevance_threshold = relevance_threshold
        self._prepare_data(df)


    def _prepare_data(self, df):

        df = df.sort_values(self.time_col)
        self.items = df.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        self.relevances = df.groupby(self.user_col)[self.relevance_col].agg(list).to_dict()
        self.user_ids = list(self.items.keys())

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):
        
        user_id = self.user_ids[idx]
        item_sequence = self.items[user_id]
        relevance_sequence = self.relevances[user_id]
        
        item_sequence = np.array(item_sequence).astype(float) 
        item_sequence = item_sequence[-self.max_length - 1:]
        relevance_sequence = relevance_sequence[-self.max_length - 1:]
        
        input_ids = item_sequence[:-1].astype(int)
        labels = item_sequence[1:].astype(int)
        skip = (np.array(relevance_sequence)[:-1] < self.relevance_threshold).astype(int)
        
        item = {'user_id': user_id, 'input_ids': input_ids,
                'labels': labels, 'skip': skip}

        return item


class CausalNegativeFeedbackDataset(Dataset):
    """Training dataset for negative feedback."""

    padding_value = 0
    labels_padding_value = -100

    def __init__(self, df, max_length,
                 relevance_col, relevance_threshold,
                 user_col, item_col='item_id',
                 time_col='time_idx', 
                 #fixed_feedback_seq_len=False
                ):
        """
        Args:
            df (pd.DataFrame): Dataframe with interactions.
            max_length (int): Maximum sequence legth.
            user_col (str): User column in dataframe.
            item_col (str): Item column in dataframe.
            time_col (str): Time column in dataframe.
            relevance_col (str): Relevance column in dataframe.
            relevance_threshold (float): Threshold for negative feedback.
        """

        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.relevance_col = relevance_col
        self.relevance_threshold = relevance_threshold
        #self.fixed_feedback_seq_len = fixed_feedback_seq_len
        self._prepare_data(df)

    def _prepare_data(self, df):

        df = df.sort_values(self.time_col)
        self.items = df.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        self.relevances = df.groupby(self.user_col)[self.relevance_col].agg(list).to_dict()
        self.user_ids = list(self.items.keys())

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):
        
        user_id = self.user_ids[idx]
        item_sequence = self.items[user_id]
        relevance_sequence = self.relevances[user_id]
        
        item_sequence = np.array(item_sequence).astype(float) 
        
        #if self.fixed_feedback_seq_len:
        positive_item_sequence = item_sequence[np.array(relevance_sequence) >= self.relevance_threshold]
        negative_item_sequence = item_sequence[np.array(relevance_sequence) < self.relevance_threshold]

        positive_item_sequence = positive_item_sequence[-self.max_length - 1:]
        negative_item_sequence = negative_item_sequence[-self.max_length - 1:]
#         else:
#             item_sequence = item_sequence[-self.max_length - 1:] 
#             relevance_sequence = relevance_sequence[-self.max_length - 1:] 

#             positive_item_sequence = item_sequence[np.array(relevance_sequence) >= self.relevance_threshold]
#             negative_item_sequence = item_sequence[np.array(relevance_sequence) < self.relevance_threshold]

        positive_input_ids = positive_item_sequence[:-1].astype(int)
        negative_input_ids = negative_item_sequence[:-1].astype(int)
        positive_labels = positive_item_sequence[1:].astype(int)
        negative_labels = negative_item_sequence[1:].astype(int)

        item = {'user_id': user_id, 'input_ids': positive_input_ids, 'negative_input_ids': negative_input_ids,
                'labels': positive_labels, 'negative_labels': negative_labels}

        return item


class CausalPredictionDataset(CausalDataset):
    """Prediction / validation dataset for negative feedback."""

    def __init__(self, df, max_length,
                 relevance_col, relevance_threshold,
                 user_col, item_col='item_id',
                 time_col='time_idx', 
                 validation_mode=False, validation_size=None, positive_eval=True):
        """
        Args:
            df (pd.DataFrame): Dataframe with interactions.
            max_length (int): Maximum sequence legth.
            user_col (str): User column in dataframe.
            item_col (str): Item column in dataframe.
            time_col (str): Time column in dataframe.
            session_col (str): Session column in dataframe.
            relevance_col (str): Relevance column in dataframe.
            relevance_threshold (float): Threshold for negative feedback.
            add_session_separator (bool): If True then add separating token between sessions.
            last_positive (bool): If True then take only users
                with last positive interaction for validation.
            validation_mode (bool): Validation or test prediction mode.
            validation_size (int): Number of users to sample in validation set.
                If None then take all users.
        """

        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.relevance_col = relevance_col
        self.relevance_threshold = relevance_threshold
        self.validation_mode = validation_mode
        self.validation_size = validation_size
        self.positive_eval = positive_eval

        users = df[self.user_col].unique()
        if validation_size and (validation_size < len(users)):
            users = np.random.choice(users, size=validation_size, replace=False)
            df = df[df[self.user_col].isin(users)]

        self._prepare_data(df)

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.items[user_id]
        relevance_sequence = self.relevances[user_id]
            
        input_ids = np.array(item_sequence).astype(float)
        positive_input_ids = input_ids[np.array(relevance_sequence) >= self.relevance_threshold].astype(int)
        
        item = {'user_id': user_id}
        
        if self.validation_mode:
            item['target'] = positive_input_ids[-1]
            item['full_history'] = item_sequence[:-1]
            
            if self.positive_eval:
                item['input_ids'] = positive_input_ids[-self.max_length - 1: -1]
            else:
                item['input_ids'] = item['full_history'][-self.max_length:]
            
        else:
            item['full_history'] = item_sequence
            
            if self.positive_eval:
                item['input_ids'] = positive_input_ids[-self.max_length:]
            else:
                item['input_ids'] = item['full_history'][-self.max_length:]

        return item    
    
    
class PaddingCollateFn:
    """Collate function for aggregating in batch."""

    def __init__(self, padding_value=0,
                 labels_padding_value=-100, labels_keys=['labels'], add_negative_mask=False):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value
        self.labels_keys = labels_keys
        self.add_negative_mask = add_negative_mask

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key in self.labels_keys:
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value

            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(values, batch_first=True,
                                               padding_value=padding_value)

        positive_attention_mask = collated_batch['input_ids'] != self.padding_value
        collated_batch['attention_mask'] = positive_attention_mask.to(dtype=torch.float32)
        
        if self.add_negative_mask:
            negative_attention_mask = collated_batch['negative_input_ids'] != self.padding_value
            collated_batch['negative_attention_mask'] = negative_attention_mask.to(dtype=torch.float32)

        return collated_batch
