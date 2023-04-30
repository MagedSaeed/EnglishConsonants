import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

from dotless_arabic.experiments.nlms.src import constants


class LitNeuralLanguageModel(LightningModule):
    def __init__(
        self,
        vocab_size,
        unk_token_id=0,
        pad_token_id=1,
        tie_weights=True,
        num_layers=constants.NUM_LAYERS,
        hidden_size=constants.HIDDEN_SIZE,
        dropout_prob=constants.DROPOUT_PROB,
        learning_rate=constants.LEARNING_RATE,
        embedding_size=constants.EMBEDDING_SIZE,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
        )
        self.gru_layer = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.first_dense_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
        )
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()
        self.second_dense_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
        )

        # weights tieing
        if tie_weights:
            assert (
                self.embedding_size == self.hidden_size
            ), "in weights tieing, embedding size should be the same as hidden size"
            self.second_dense_layer.weight = self.embedding_layer.weight

    def forward(self, x, hiddens=None):
        outputs = self.embedding_layer(x)
        if hiddens is None:
            outputs, hiddens = self.gru_layer(outputs)
        else:
            outputs, hiddens = self.gru_layer(outputs, hiddens)
        outputs = self.first_dense_layer(outputs)
        outputs = self.dropout_layer(outputs)
        outputs = self.relu(outputs)
        outputs = self.second_dense_layer(outputs)
        return outputs, hiddens

    # def _loss_reduced(self, batch_outputs, batch_labels, ignore_oovs=False):
    #     """
    #     This method computes the loss for a sentence
    #     without special tokens such as PAD and optionally UNK

    #     batch_outputs shape: [sequence_length:embd_size]
    #     batch_labels shape: [sequence_length]

    #     if the whole sentence labels turns out to be a PAD/UNK, it returns None instead.
    #     """

    #     # https://discuss.pytorch.org/t/ignore-padding-area-in-loss-computation/95804
    #     if ignore_oovs:
    #         # https://discuss.pytorch.org/t/when-to-use-ignore-index/5935/11?u=magedsaeed
    #         batch_labels[batch_labels == self.unk_token_id] = self.pad_token_id
    #     # print('batch labels before',batch_labels)
    #     mask = batch_labels != self.pad_token_id
    #     batch_outputs = batch_outputs[mask]
    #     batch_labels = batch_labels[mask]
    #     # print('batch labels after',batch_labels)
    #     if len(batch_labels) == 0:
    #         return None
    #     loss = F.cross_entropy(batch_outputs,batch_labels)
    #     # # substitute those uneeded tokens with zero loss
    #     # loss_mask = torch.isin(
    #     #     batch_labels,
    #     #     torch.Tensor(ignored_ids).to(DEVICE),
    #     #     invert=True,
    #     # )
    #     # loss = loss.where(
    #     #     loss_mask,
    #     #     torch.tensor(0.0).to(DEVICE),
    #     # )
    #     return loss

    def _get_loss(self, batch, ignore_oovs=False, loss_reduction="mean"):
        inputs, labels = batch
        outputs, hiddens = self(inputs)
        # https://discuss.pytorch.org/t/cross-entropy-loss-for-a-sequence-time-series-of-output/4309
        outputs = outputs.view(-1, self.vocab_size)
        labels = labels.view(-1)
        # losses = [
        #     self._loss_reduced(
        #         batch_outputs = batch_outputs,
        #         batch_labels = batch_labels,
        #         ignore_oovs = ignore_oovs,
        #     )
        #     for batch_outputs,batch_labels in zip(outputs,labels)
        # ]
        # losses = torch.stack([loss for loss in losses if loss is not None])
        if ignore_oovs:
            # https://discuss.pytorch.org/t/when-to-use-ignore-index/5935/11?u=magedsaeed
            labels[labels == self.unk_token_id] = self.pad_token_id
        loss = F.cross_entropy(
            outputs,
            labels,
            ignore_index=self.pad_token_id,
            reduction=loss_reduction,
        )
        # loss = losses.mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return {'val_loss':loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.1,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
