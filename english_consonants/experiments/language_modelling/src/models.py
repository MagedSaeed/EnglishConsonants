import math
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

from english_consonants.experiments.language_modelling.src import constants


class LitNeuralLanguageModel(LightningModule):
    def __init__(
        self,
        vocab_size,
        unk_token_id=0,
        pad_token_id=1,
        tie_weights=True,
        model_type="lstm",
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
        # self.gru_layer =
        if model_type.lower() == "lstm".lower():
            self.rnn = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout_prob,
                batch_first=True,
            )
        elif model_type.lower() == "gru".lower():
            self.rnn = nn.GRU(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout_prob,
                batch_first=True,
            )
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()
        self.dense_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
        )

        # weights tieing
        if tie_weights:
            assert (
                self.embedding_size == self.hidden_size
            ), "in weights tieing, embedding size should be the same as hidden size"
            self.dense_layer.weight = self.embedding_layer.weight

    def forward(self, x, hiddens=None):
        outputs = self.embedding_layer(x)
        # adding dropout to embeddings
        outputs = self.dropout_layer(outputs)
        if hiddens is None:
            outputs, hiddens = self.rnn(outputs)
        else:
            outputs, hiddens = self.rnn(outputs, hiddens)
        # outputs = self.first_dense_layer(outputs)
        outputs = self.dropout_layer(outputs)
        outputs = self.relu(outputs)
        outputs = self.dense_layer(outputs)
        return outputs, hiddens

    def step(self, batch, ignore_oovs=False, loss_reduction="mean"):
        inputs, labels = batch
        outputs, hiddens = self(inputs)
        # https://discuss.pytorch.org/t/cross-entropy-loss-for-a-sequence-time-series-of-output/4309
        outputs = outputs.view(-1, self.vocab_size)
        labels = labels.view(-1)
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
        loss = self.step(batch)
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerLanguageModel(LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
        pad_token_id=1,
        unk_token_id=1,
        learning_rate=constants.LEARNING_RATE,
    ):
        super(TransformerLanguageModel, self).__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=512,
            dropout=dropout,
        )
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(embed_dim, self.vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def step(self, batch, ignore_oovs=False, loss_reduction="mean"):
        inputs, labels = batch
        src_mask = src_mask = self.generate_square_subsequent_mask(inputs.size(0))
        # src_mask = None  which one do I use?
        outputs = self(inputs, src_mask)
        outputs = outputs.view(-1, self.vocab_size)
        labels = labels.view(-1)
        if ignore_oovs:
            print("performing a step, ignoring OOVs")
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
        loss = self.step(batch)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            1.0,
            gamma=0.95,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.learning_rate,
    #     )
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer=optimizer,
    #         factor=0.5,
    #         patience=1,
    #         verbose=True,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": scheduler,
    #         "monitor": "val_loss",
    #     }
