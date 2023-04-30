import os

from pytorch_lightning import seed_everything

import wandb
from english_consonants.experiments.language_modelling.src import constants


def configure_environment():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    seed_everything(constants.RANDOM_SEED, workers=True)
    wandb.login()
