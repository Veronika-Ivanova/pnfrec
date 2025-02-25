import logging
import os
import random
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
import torch
from torch.utils.data import DataLoader

from datasets import (
    PaddingCollateFn,
    CausalNegativeFeedbackDataset,
    CausalPredictionDataset,
)
from modules import SeqRecPosNegContrastive
from models import SASRecPosNegContrastive
from postprocess import preds2recs
from preprocess import add_time_idx, prepare_splitted_data
from metrics import compute_replay_metrics
from utils import extract_validation_history, fix_seed

logging.getLogger().setLevel(logging.ERROR)

cfg = OmegaConf.load("negative_feedback/runs/configs/SASRec_ml.yaml")

os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.cuda_visible_devices}"

PROJECT_PATH = f"{cfg.project_path}"
DATA_PATH = f"{PROJECT_PATH}{cfg.data_path}"

USER_COL = f"{cfg.dataset.user_col}"
ITEM_COL = f"{cfg.dataset.item_col}"
RELEVANCE_COL = f"{cfg.dataset.relevance_col}"
RELEVANCE_THRESHOLD = cfg.dataset.relevance_threshold
MAX_LENGTH = round(cfg.dataset.max_length / 2)

VALIDATION_SIZE = cfg.dataloader.validation_size
BATCH_SIZE = cfg.dataloader.batch_size
TEST_BATCH_SIZE = cfg.dataloader.test_batch_size
NUM_WORKERS = cfg.dataloader.num_workers

dropout = cfg.model.dropout
hidden_units = cfg.model.hidden_units
lr = cfg.model.lr
num_blocks = cfg.model.num_blocks
neg_CE_coef = cfg.model.neg_CE_coef
contrastive_coef = cfg.model.contrastive_coef


fix_seed()
train, train_full, validation, test_pos, test_neg, last_item_pos, last_item_neg = prepare_splitted_data(
    DATA_PATH,
    user_col=USER_COL,
    relevance_col=RELEVANCE_COL,
    filter_negative=False,
    relevance_threshold=RELEVANCE_THRESHOLD,
)


def get_eval_dataset(validation, validation_size=VALIDATION_SIZE):
    validation_users = validation.user_id.unique()

    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(
            validation_users, size=validation_size, replace=False
        )

    eval_dataset = CausalPredictionDataset(
        validation[validation.user_id.isin(validation_users)],
        max_length=MAX_LENGTH,
        relevance_col=RELEVANCE_COL,
        relevance_threshold=RELEVANCE_THRESHOLD,
        user_col=USER_COL,
        validation_mode=True,
        positive_eval=True, 
    )

    return eval_dataset


train_dataset = CausalNegativeFeedbackDataset(
    train,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
)
eval_dataset = get_eval_dataset(validation)

collate_fn_train = PaddingCollateFn(add_negative_mask=True, labels_keys=['labels', 'negative_labels'])
collate_fn_val = PaddingCollateFn()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_train,
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
)

predict_pos_dataset = CausalPredictionDataset(
    test_pos,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
    positive_eval=True,
)
predict_neg_dataset = CausalPredictionDataset(
    test_neg,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
    positive_eval=True,
)
predict_pos_loader = DataLoader(
    predict_pos_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
)
predict_neg_loader = DataLoader(
    predict_neg_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
)

item_count = train.item_id.max()
add_head = True

@hydra.main(version_base=None, config_path="configs", config_name="SASRec_ml")
def main(cfg):

    fix_seed()
    model = SASRecPosNegContrastive(
        item_num=item_count,
        add_head=add_head,
        maxlen=MAX_LENGTH,
        num_heads=1,
        dropout_rate=dropout,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
    )

    fix_seed()
    seqrec_module = SeqRecPosNegContrastive(model, 
                                            lr=lr, 
                                            predict_top_k=10, 
                                            filter_seen=True, 
                                            neg_CE_coef=neg_CE_coef, 
                                            contrastive_coef=contrastive_coef)
    early_stopping = EarlyStopping(
        monitor="val_ndcg", mode="max", patience=cfg.patience, verbose=False
    )

    model_summary = ModelSummary(max_depth=2)
    checkpoint = ModelCheckpoint(
        save_top_k=1, monitor="val_ndcg", mode="max", save_weights_only=True
    )
    callbacks = [early_stopping, model_summary, checkpoint]

    trainer = pl.Trainer(
        callbacks=callbacks,
        enable_checkpointing=True,
        gpus=1,
        max_epochs=cfg.trainer_params.max_epochs,
        deterministic=True,
    )

    trainer.fit(
        model=seqrec_module,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
    )

    seqrec_module.load_state_dict(
        torch.load(checkpoint.best_model_path)["state_dict"]
    )
    history = extract_validation_history(trainer.logger.experiment.log_dir)
    val_metrics = {
        "val_ndcg": history["val_ndcg"].max(),
        "val_hit_rate": history["val_hit_rate"].max(),
        "val_mrr": history["val_mrr"].max(),
    }

    seqrec_module.predict_top_k = 10
    seqrec_module.filter_seen = True
    preds_pos = trainer.predict(model=seqrec_module, dataloaders=predict_pos_loader)
    preds_neg = trainer.predict(model=seqrec_module, dataloaders=predict_neg_loader)
    recs_pos = preds2recs(preds_pos)
    recs_neg = preds2recs(preds_neg)

    metrics = compute_replay_metrics(
        last_item_pos,
        last_item_neg,
        recs_pos,
        recs_neg,
        train_full,
        relevance_col=RELEVANCE_COL,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )
    
    print(metrics)


if __name__ == "__main__":

    main()
