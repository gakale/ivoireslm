import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tokenizer.character import CharacterTokenizer


def test_character_vocab_is_deterministic_and_sorted():
    first = CharacterTokenizer.train("éa\na")
    second = CharacterTokenizer.train("aé\n")
    assert first.id_to_token == second.id_to_token
    assert first.id_to_token[4:] == ("\n", "a", "é")


def test_character_tokenizer_roundtrip_with_boundaries():
    tokenizer = CharacterTokenizer.train("Bonjour Côte d’Ivoire !\n")
    text = "Côte d’Ivoire !\n"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.decode(ids, skip_special=True) == text


def test_character_tokenizer_maps_unseen_character_to_unknown():
    tokenizer = CharacterTokenizer.train("abc")
    ids = tokenizer.encode("abz")
    assert ids[-1] == tokenizer.token_to_id["<UNK>"]
    assert tokenizer.decode(ids).endswith("<UNK>")


def test_character_tokenizer_rejects_invalid_id():
    tokenizer = CharacterTokenizer.train("abc")
    try:
        tokenizer.decode([tokenizer.vocab_size])
    except ValueError as error:
        assert "invalide" in str(error)
    else:
        raise AssertionError("un identifiant invalide doit être rejeté")
