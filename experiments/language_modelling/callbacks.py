from pytorch_lightning import Callback
from collections import defaultdict


class LossMetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.losses = defaultdict(list)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if batch_idx % trainer.log_every_n_steps == 0:
            self.losses["training"].append(outputs["loss"])

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if batch_idx % trainer.log_every_n_steps == 0:
            self.losses["validation"].append(outputs["val_loss"])
