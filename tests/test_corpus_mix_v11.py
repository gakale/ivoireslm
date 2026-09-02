import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_corpus_mix_v11", ROOT / "scripts/data/audit_corpus_mix_v11.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CONFIG = json.loads((ROOT / "configs/corpus_mix_v11_candidate.json").read_text())


def test_target_weights_are_normalized_and_natural_language_dominates():
    weights = CONFIG["target_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["mathematics_reasoning"] <= 0.01
    assert (
        weights["ivoireslm_corpus_v0.9.0_base"]
        + weights["natural_french_open"]
        + weights["natural_french_conversation_open"]
        + weights["natural_ivoirian_conversation_verified"]
    ) >= 0.9


def test_current_math_heavy_supplement_is_rejected():
    report = {
        "dataset_id": "current",
        "characters": 108_554_194,
        "domain_characters": {
            "mathematics_reasoning": 97_428_589,
            "natural_french_open": 3_110_725,
            "natural_english_open": 3_309_385,
            "code_agents": 3_722_158,
            "cybersecurity_defensive": 983_337,
        },
    }
    audit = MODULE.compute_audit(report, CONFIG)
    assert audit["training_authorized"] is False
    assert audit["domain_shares"]["mathematics_reasoning"] > 0.89
    assert "maximum_mathematics_share" in audit["failed_checks"]
    assert "minimum_ivoirian_conversation_characters" in audit["failed_checks"]


def test_balanced_licensed_supplement_passes():
    report = {
        "dataset_id": "balanced",
        "characters": 50_000_000,
        "domain_characters": {
            "natural_french_open": 30_000_000,
            "natural_french_conversation_open": 2_000_000,
            "natural_ivoirian_conversation_verified": 6_000_000,
            "natural_english_open": 5_000_000,
            "ivoirian_languages_verified": 4_000_000,
            "mathematics_reasoning": 1_000_000,
            "code_agents": 1_000_000,
            "cybersecurity_defensive": 1_000_000,
        },
    }
    audit = MODULE.compute_audit(report, CONFIG)
    assert audit["training_authorized"] is True
    assert audit["failed_checks"] == []
    assert audit["test_opened"] is False
