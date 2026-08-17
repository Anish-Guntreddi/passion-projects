"""Small shared constants with no dependencies on other forgelm modules.

Kept separate from ``forgelm.tokenizer`` and ``forgelm.config`` so those two
modules never need to import each other just to share these values.
"""

from __future__ import annotations

#: Special tokens every tokenizer instance reserves, in a fixed order.
#: Their token ids are always the last ``len(DEFAULT_SPECIAL_TOKENS)`` ids in
#: the vocabulary (see docs/decisions/0002-tokenizer-scope.md).
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<|bos|>", "<|eos|>", "<|pad|>")

#: Byte-level BPE always starts from the 256 raw byte values.
BASE_BYTE_VOCAB_SIZE = 256
