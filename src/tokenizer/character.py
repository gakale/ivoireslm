from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")


@dataclass(frozen=True)
class CharacterTokenizer:
    id_to_token: tuple[str, ...]

    @classmethod
    def train(cls, text: str) -> "CharacterTokenizer":
        characters = tuple(sorted(set(text), key=ord))
        return cls(SPECIAL_TOKENS + characters)

    @classmethod
    def from_characters(cls, characters) -> "CharacterTokenizer":
        return cls(SPECIAL_TOKENS + tuple(sorted(set(characters), key=ord)))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        mapping = self.token_to_id
        ids = [mapping["<BOS>"]] if add_bos else []
        unknown = mapping["<UNK>"]
        ids.extend(mapping.get(character, unknown) for character in text)
        if add_eos:
            ids.append(mapping["<EOS>"])
        return ids

    def decode(self, ids, *, skip_special: bool = False) -> str:
        pieces = []
        for token_id in ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"identifiant de token invalide : {token_id}")
            token = self.id_to_token[token_id]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def character_frequencies(self, text: str) -> dict[str, int]:
        return dict(sorted(Counter(text).items(), key=lambda item: ord(item[0])))

