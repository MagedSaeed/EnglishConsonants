import io
import tempfile
import warnings
import tkseem as tk
import sentencepiece as spm
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


# class CharacterTokenizer(tk.CharacterTokenizer):
#     @classmethod
#     @lru_cache(maxsize=10_000)
#     def split_text(cls, text):
#         splitted_text = []
#         for character in list(text):
#             if character.isspace():
#                 splitted_text.append("<##>")
#             else:
#                 splitted_text.append(character)
#         return splitted_text

#     def detokenize(self, tokens):
#         """Convert tokens to a string

#         Args:
#             tokens (list): list of tokens

#         Returns:
#             str: detokenized string
#         """
#         detokenized = "".join(tokens).replace("<##>", " ")
#         return detokenized

#     def train(self, text=None, file_path=None):
#         print("Training CharacterTokenizer ...")

#         assert (
#             file_path is not None or text is not None
#         ), "either file_path or text should be provided."

#         if not text:
#             text = open(file_path, "r").read()

#         tokens_frequency = defaultdict(int)
#         for word in WordTokenizer.split_text(text):
#             tokens_frequency[word] += 1

#         self.vocab = self._truncate_dict(dict(tokens_frequency))
#         self.vocab = tokenize_vocab(vocab=self.vocab, tokenizer=self)
#         self.vocab_size = len(self.vocab)


class SentencePieceTokenizer(tk.SentencePieceTokenizer):
    """Sentencepiece based tokenization."""

    def train(
        self,
        text=None,
        file_path=None,
        model_type="bpe",
        **kwargs,
    ):
        """Train using sentence piece

        Args:
            file_path (str): file to train
            model_type (str, optional): train using sp. Defaults to "bpe".
        """
        assert (
            text is not None or file_path is not None
        ), "file_path or text should be provided"

        if file_path:
            text = open(file_path, "r").read().splitlines()

        text_file = tempfile.NamedTemporaryFile()

        with open(text_file.name, "w") as file:
            file.write(text)

        print("Training SentencePiece ...")
        self.model = io.BytesIO()

        spm.SentencePieceTrainer.train(
            input=text_file.name,
            model_writer=self.model,
            vocab_size=self.vocab_size,
            model_type=model_type,
            character_coverage=kwargs.get("character_coverage", 1.0),
            unk_id=0,
            pad_id=1,
            bos_id=kwargs.get("bos_id", -1),
            eos_id=kwargs.get("eos_id", -1),
            user_defined_symbols=self.special_tokens,
            normalization_rule_name="identity",
            minloglevel=1,  # to suppress train logs, https://github.com/speechbrain/speechbrain/pull/206#issuecomment-669260984
        )
        model_file = tempfile.NamedTemporaryFile()
        self.save_model(model_file.name)
        self.sp = spm.SentencePieceProcessor(model_file=model_file.name)
        self.vocab_size = self.sp.vocab_size()
        self.vocab = {
            self.sp.id_to_piece(id): id for id in range(self.sp.get_piece_size())
        }

    def split_text(self, text):
        """
        For this tokenizer, split text is the same as tokenize
        """
        warnings.warn("sentencepiece tokenizer cannot split text unless with PBE mode")
        return self.tokenize(text)

    def tokenize_from_splits(self, text):
        """
        For this tokenizer, split text is the same as tokenize
        """
        warnings.warn("sentencepiece tokenizer cannot split text unless with PBE mode")
        return self.tokenize(text)


TOKENIZERS_MAP = {
    tokenizer_class.__name__: tokenizer_class
    for tokenizer_class in [
        WordTokenizer,
        SentencePieceTokenizer,
    ]
}
