import torch

from dotless_arabic.experiments.constants import *


NUM_LAYERS = 4  # for GRU

MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128 
HIDDEN_SIZE = 512
DROPOUT_PROB = 0.333
EMBEDDING_SIZE = 512
LEARNING_RATE = 0.001
DEFAULT_VOCAB_COVERAGE = 0.95  # for tokenizer to consider only tokens that cover this value of the running text
SEQUENCE_LENGTH_PERCENTILE = 0.99  # percentile to consider lengths and return up to that percentile. 0.5 percentile should return the average
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIGHTING_ACCELERATOR = "auto"  # used by pytorch_lightning

TEST_SIZE = 0.1
VAL_SIZE = 0.05

GPU_DEVICES = "0"  # consider only one CPU core if there is no GPU

CPU_DEVICES = 1  # I tried with higher values but this did not work
