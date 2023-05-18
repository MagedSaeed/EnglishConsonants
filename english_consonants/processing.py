from collections import defaultdict
from functools import lru_cache
import math
import re
import string
from tqdm.auto import tqdm

ENGLISH_LETTERS = string.ascii_lowercase


def process_english(text):
    # add spaces between punctuations, if there is not
    # text = text.lower()
    # text = re.sub(
    #     r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>`؛=+\-\*\&\^\%\$\#\@\!])""",
    #     r" \1 ",
    #     text,
    # )
    # # remove any non arabic character
    # text = "".join(
    #     [c for c in text if c in ENGLISH_LETTERS or c.isspace()]
    # )  # keep only english chars and spaces
    # text = re.sub("\s{2,}", " ", text).strip()  # remove multiple spaces
    # """
    #   interestingly, there is a difference betwen re.sub('\s+',' ',s) and re.sub('\s{2,}',' ',s)
    #   the first one remove newlines while the second does not.
    # """
    return text.strip()


def mask_vowels(text, mask=""):
    text_with_no_vowels = re.sub(
        r"[AEIOU]",
        mask,
        text,
        flags=re.IGNORECASE,
    )
    return text_with_no_vowels


@lru_cache()
def tokens_frequency(dataset, use_tqdm=True):
    frequencies = defaultdict(int)
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        for token in document.split():
            frequencies[token] += 1
    frequencies = dict(frequencies)
    return frequencies


def calculate_entropy(tokens_frequency):
    # https://stackoverflow.com/q/43419803/4412324
    # https://stackoverflow.com/a/40496783/4412324
    total_number_of_tokens = sum(tokens_frequency.values())
    entropy = -sum(
        (word_frequency / total_number_of_tokens)
        * math.log2(word_frequency / total_number_of_tokens)
        for word_frequency in tokens_frequency.values()
    )
    return entropy
