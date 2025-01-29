import logging
import os
import random
import sys

sys.path.append("/home/jovyan/ivanova/negative_feedback/src")
sys.path.remove("/home/jovyan/.imgenv-vasilyev-0/lib/python3.7/site-packages")

import hydra
import mlflow
import numpy as np
from omegaconf import OmegaConf
import optuna
import pandas as pd
import pickle
from pprint import pformat
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
import torch
from torch.utils.data import DataLoader

from datasets import (
    PaddingCollateFn,
    CausalDataset,
    CausalNegativeFeedbackDataset,
    CausalPredictionDataset,
)
from modules import SeqRec, SeqRecContrastiveBaseline
from models import SASRec, SASRecContrastiveBaseline
from postprocess import preds2recs
from preprocess import add_time_idx, prepare_splitted_data
from metrics import compute_replay_metrics
from utils import extract_validation_history

logging.getLogger().setLevel(logging.ERROR)

pl.seed_everything(42, workers=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cfg = OmegaConf.load("negative_feedback/src/configs/SASRec_ml.yaml")

os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.cuda_visible_devices}"

PROJECT_PATH = "~/ivanova/negative_feedback/"
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

mlflow.set_tracking_uri(f"{cfg.ml_flow.set_tracking_uri}")
mlflow.set_experiment(f"{cfg.ml_flow.set_experiment}")


train, validation, test, last_item = prepare_splitted_data(
    DATA_PATH,
    user_col=USER_COL,
    relevance_col=RELEVANCE_COL,
    filter_negative=True, ######
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
        positive_eval=True, ######
    )

    return eval_dataset


train_dataset = CausalDataset(
    train,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
)
eval_dataset = get_eval_dataset(validation)

collate_fn_train = PaddingCollateFn()
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

predict_dataset = CausalPredictionDataset(
    test,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
    positive_eval=True, ######
)
predict_loader = DataLoader(
    predict_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
)

item_count = train.item_id.max()
add_head = True


# Obtain hyperparameters for this trial
def suggest_hyperparameters(trial):
    lr = trial.suggest_categorical("lr", [5e-4, 1e-3])
    dropout = trial.suggest_float("dropout", 0.05, 0.5, step=0.05)
    hidden_units = trial.suggest_int("hidden_units", 32, 512, log=True)
    num_blocks = trial.suggest_int("num_blocks", 1, 3)
    print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
    return lr, dropout, hidden_units, num_blocks


def objective(trial):
    print("\n********************************\n")

    # Start a new mlflow run
    with mlflow.start_run(run_name="ml1m_sasrec_positive_items"):
        # Get hyperparameter suggestions created by optuna and log them as params using mlflow
        lr, dropout, hidden_units, num_blocks = suggest_hyperparameters(trial)
        mlflow.log_params(trial.params)
        mlflow.log_params({"maxlen": MAX_LENGTH, "batch_size": BATCH_SIZE})

        model = SASRec(
            item_num=item_count,
            add_head=add_head,
            maxlen=MAX_LENGTH,
            num_heads=1,
            dropout_rate=dropout,
            hidden_units=hidden_units,
            num_blocks=num_blocks,
        )

        seqrec_module = SeqRec(model, lr=lr, predict_top_k=10, filter_seen=True)
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
        mlflow.log_metrics(val_metrics)

        seqrec_module.predict_top_k = 10
        seqrec_module.filter_seen = True
        preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)
        recs = preds2recs(preds)

        metrics = compute_replay_metrics(
            last_item,
            recs,
            train,
            relevance_col=RELEVANCE_COL,
            relevance_threshold=RELEVANCE_THRESHOLD,
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({"n_epoch": history.epoch.max()})
        mlflow.end_run()

    # Return the best validation loss achieved by the network.
    # This is needed as Optuna needs to know how the suggested hyperparameters are influencing the network loss.
    return val_metrics["val_ndcg"]


@hydra.main(version_base=None, config_path="configs", config_name="SASRec_ml")
def main(cfg):

    pl.seed_everything(42, workers=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create the optuna study which shares the experiment name
    study = optuna.create_study(
        study_name="pytorch-mlflow-optuna", direction="maximize"
    )
    study.optimize(objective, n_trials=50)

    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  val_ndcg: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":

    main()
