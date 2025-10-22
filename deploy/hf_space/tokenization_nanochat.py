"""
Hugging Face tokenizer wrapper for nanochat's rustbpe+tiktoken vocabulary.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import tiktoken
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer

from configuration_nanochat import NanoChatConfig

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


class NanoChatTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"tokenizer_file": "tokenizer/tokenizer.pkl"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file: Optional[str] = None, **kwargs):
        if tokenizer_file is None:
            raise ValueError("tokenizer_file must be provided")

        with open(tokenizer_file, "rb") as handle:
            self._encoding: tiktoken.Encoding = pickle.load(handle)

        self._id_to_token: List[str] = [self._encoding.decode([i]) for i in range(self._encoding.n_vocab)]
        self.vocab: Dict[str, int] = {token: idx for idx, token in enumerate(self._id_to_token)}

        super().__init__(
            bos_token="<|bos|>",
            eos_token="<|bos|>",
            unk_token="<|bos|>",
            pad_token="<|bos|>",
            **kwargs,
        )

        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.pad_token_id = self.vocab[self.pad_token]

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        token_ids = self._encoding.encode_ordinary(text)
        return [self._id_to_token[token_id] for token_id in token_ids]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token[index]

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            raise ValueError("nanochat tokenizer only supports single sequences")
        return [self.bos_token_id] + token_ids_0

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        del token_ids_1
        return [0] * (len(token_ids_0) + 1)  # +1 for BOS

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        target_dir = os.path.join(save_directory, "tokenizer")
        os.makedirs(target_dir, exist_ok=True)
        filename = (filename_prefix + "-" if filename_prefix else "") + "tokenizer.pkl"
        dest_file = os.path.join(target_dir, filename)
        with open(dest_file, "wb") as handle:
            pickle.dump(self._encoding, handle)
        return (dest_file,)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        del clean_up_tokenization_spaces, spaces_between_special_tokens, kwargs
        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if tid not in self.all_special_ids]
        return self._encoding.decode(token_ids)


# Register the tokenizer so AutoTokenizer can locate it via NanoChatConfig.
AutoTokenizer.register(NanoChatConfig, NanoChatTokenizer)
