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


OASST = load("snapshot_openassistant_fr_v01", "scripts/data/snapshot_openassistant_fr_v01.py")
WIKI = load("snapshot_wikipedia_fr_natural_v03", "scripts/data/snapshot_wikipedia_fr_natural_v03.py")
BUILD = load("build_natural_french_supplement_v11", "scripts/data/build_natural_french_supplement_v11.py")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_wikipedia_split_has_no_test_and_is_deterministic():
    assert WIKI.stable_split(12345) == WIKI.stable_split(12345)
    assert WIKI.stable_split(12345) in {"train", "validation"}


def test_oasst_filter_rejects_synthetic_and_keeps_reviewed_french():
    row = {
        "lang": "fr", "deleted": False, "synthetic": False, "review_result": True,
        "text": "Bonjour, voici une réponse humaine suffisamment longue.",
        "message_id": "answer", "parent_id": "question", "message_tree_id": "tree",
        "role": "assistant", "review_count": 2,
    }
    record, reason = OASST.accepted_record(row, "train")
    assert reason is None
    assert record["domain"] == "natural_french_conversation_open"
    row["synthetic"] = True
    assert OASST.accepted_record(row, "train")[1] == "deleted_or_synthetic"


def test_builder_pairs_dialogue_and_requires_consent(tmp_path):
    wiki, oasst = tmp_path / "wiki", tmp_path / "oasst"
    write_jsonl(wiki / "pages.jsonl", [{
        "page_id": 1, "url": "https://example.test/wiki/1", "title": "Culture",
        "attribution": "Auteurs", "split": "train", "text": "Texte naturel français. " * 10,
    }])
    base = {
        "group_id": "oasst1-tree:t", "source_url": "https://example.test/oasst",
        "split": "train", "license": "Apache-2.0",
    }
    write_jsonl(oasst / "train.jsonl", [
        {**base, "message_id": "q", "parent_id": None, "role": "prompter", "text": "Comment vas-tu ?"},
        {**base, "message_id": "a", "parent_id": "q", "role": "assistant", "text": "Je vais bien, merci."},
    ])
    write_jsonl(oasst / "validation.jsonl", [])
    consent = tmp_path / "consent.jsonl"
    write_jsonl(consent, [
        {"id": "ok", "split": "validation", "prompt": "On dit quoi ?", "response": "On est ensemble.",
         "license": "CC0-1.0", "consent_for_training": True, "no_personal_data": True},
        {"id": "bad", "split": "train", "prompt": "Privé", "response": "Non",
         "license": "CC0-1.0", "consent_for_training": False, "no_personal_data": True},
    ])
    report = BUILD.build(wiki, oasst, tmp_path / "output", consent)
    assert report["test_created"] is False
    assert report["consented_ivoirian_rejections"] == {"missing_training_consent": 1}
    assert report["domain_characters"]["natural_ivoirian_conversation_verified"] > 0
    combined = (tmp_path / "output/train.jsonl").read_text() + (tmp_path / "output/validation.jsonl").read_text()
    assert "Utilisateur : Comment vas-tu ?" in combined
    assert "Privé" not in combined


def test_builder_accepts_standardized_ivoirian_open_data(tmp_path):
    wiki, oasst, grounded = tmp_path / "wiki", tmp_path / "oasst", tmp_path / "grounded"
    write_jsonl(wiki / "pages.jsonl", [{
        "page_id": 2, "url": "https://example.test/wiki/2", "title": "Économie",
        "attribution": "Auteurs", "split": "train", "text": "Texte français naturel. " * 10,
    }])
    write_jsonl(oasst / "train.jsonl", [])
    write_jsonl(oasst / "validation.jsonl", [])
    write_jsonl(grounded / "documents.jsonl", [{
        "document_id": "ci:1", "group_id": "ci:dataset:1", "source_id": "data-gouv-ci",
        "source_url": "https://data.gouv.ci/datasets/exemple", "title": "Exemple",
        "language": "fr-CI", "domain": "natural_ivoirian_grounded_verified",
        "content_type": "open_government_dataset_description_and_rows",
        "license": "Licence Ouverte / Open Licence", "rights_tier": "A_REDISTRIBUTABLE",
        "split": "validation", "text": "Information publique ivoirienne vérifiée et attribuée.",
    }])
    report = BUILD.build(
        wiki, oasst, tmp_path / "output", ivoirian_open_data=grounded,
        dataset_id="candidate-v1.1.1",
    )
    assert report["dataset_id"] == "candidate-v1.1.1"
    assert report["domain_characters"]["natural_ivoirian_grounded_verified"] > 0
    assert report["group_leaks"] == 0
