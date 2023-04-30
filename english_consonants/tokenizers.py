import re
import tkseem as tk
from functools import lru_cache
from collections import defaultdict


def tokenize_vocab(vocab, tokenizer):
    tokenized_vocab = defaultdict(int)
    for vocab, frequency in vocab.items():
        if vocab in tokenizer.special_tokens + [
            tokenizer.pad_token,
            tokenizer.unk_token,
        ]:
            tokenized_vocab[vocab] = frequency
            continue
        subwords = tokenizer.split_text(vocab)
        for subword in subwords:
            tokenized_vocab[subword] += frequency
    return dict(tokenized_vocab)


class WordTokenizer(tk.WordTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text):
        return text.split()

    def train(self, text=None, file_path=None):
        """Train using words' frequency

        Args:
            file_path (str): file to train
        """

        print("Training WordTokenizer ...")
        self.vocab = self._truncate_dict(
            self._get_tokens_frequency(text=text, file_path=file_path)
        )
        self.vocab_size = len(self.vocab)


class CharacterTokenizer(tk.CharacterTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text):
        splitted_text = []
        for character in list(text):
            if character.isspace():
                splitted_text.append("<##>")
            else:
                splitted_text.append(character)
        return splitted_text

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("<##>", " ")
        return detokenized

    def train(self, text=None, file_path=None):
        print("Training CharacterTokenizer ...")

        assert (
            file_path is not None or text is not None
        ), "either file_path or text should be provided."

        if not text:
            text = open(file_path, "r").read()

        tokens_frequency = defaultdict(int)
        for word in WordTokenizer.split_text(text):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab = tokenize_vocab(vocab=self.vocab, tokenizer=self)
        self.vocab_size = len(self.vocab)


TOKENIZERS_MAP = {
    tokenizer_class.__name__: tokenizer_class
    for tokenizer_class in [
        WordTokenizer,
        CharacterTokenizer,
    ]
}
