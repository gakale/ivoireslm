import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOKENIZE = load("tokenize_corpus_supplement_v10", "scripts/tokenizer/tokenize_corpus_supplement_v10.py")
TRAIN = load("continue_pretraining_mix_v10_17m", "scripts/training/continue_pretraining_mix_v10_17m.py")


def test_source_buckets_cover_all_admitted_sources():
    assert set(TOKENIZE.BUCKETS) == {
        "wikipedia_fr", "wikipedia_en", "gsm8k_train", "aqua_train_dev",
        "deepseek_harness", "cisa_kev",
    }


def test_training_weights_match_documented_mix():
    config = json.loads((ROOT / "configs/corpus_mix_v10.json").read_text())
    assert TRAIN.WEIGHTS == {
        "base_v09": config["weights"]["ivoireslm_corpus_v0.9.0_base"],
        **{key: value for key, value in config["weights"].items() if key != "ivoireslm_corpus_v0.9.0_base"},
    }
    assert abs(sum(TRAIN.WEIGHTS.values()) - 1.0) < 1e-12


def test_continuation_schedule_is_bounded():
    values = [TRAIN.learning_rate(step) for step in range(TRAIN.MAX_STEPS)]
    assert min(values) >= TRAIN.MINIMUM_LEARNING_RATE
    assert max(values) <= TRAIN.LEARNING_RATE
    assert values[0] < values[TRAIN.WARMUP_STEPS - 1]
    assert values[-1] < values[TRAIN.WARMUP_STEPS]
