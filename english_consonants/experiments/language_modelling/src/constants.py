import os
import torch
from english_consonants.tokenizers import TOKENIZERS_MAP, WordTokenizer


NUM_LAYERS = 4  # for GRU

MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
HIDDEN_SIZE = 512
DROPOUT_PROB = 0.333
EMBEDDING_SIZE = 512
LEARNING_RATE = 0.001
DEFAULT_VOCAB_COVERAGE = 0.95  # for tokenizer to consider only tokens that cover this value of the running text
SEQUENCE_LENGTH_PERCENTILE = 0.95  # percentile to consider lengths and return up to that percentile. 0.5 percentile should return the average
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIGHTING_ACCELERATOR = "auto"  # used by pytorch_lightning
RNN_TYPE = "lstm"

CPU_COUNT = os.cpu_count()
NEW_LINE = "\n"
RANDOM_SEED = 42

DEFAULT_TOKENIZER_CLASS = WordTokenizer

TEST_SIZE = 0.1
VAL_SIZE = 0.05

GPU_DEVICES = "0"  # consider only one CPU core if there is no GPU

CPU_DEVICES = 1  # I tried with higher values but this did not work
