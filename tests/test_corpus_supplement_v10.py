import importlib.util
import json
import sys
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/build_corpus_supplement_v10.py"
SPEC = importlib.util.spec_from_file_location("build_corpus_supplement_v10", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_sources(root):
    long_fr = "Texte encyclopédique français propre et utile. " * 8
    long_en = "Useful open English encyclopedic text. " * 8
    write_jsonl(root / "wiki/wikipedia_fr_documents.jsonl", [{"id": 1, "title": "FR", "text": long_fr}])
    write_jsonl(root / "wiki/wikipedia_en_documents.jsonl", [{"id": 2, "title": "EN", "text": long_en}])
    write_jsonl(root / "math/GSM8K_train.jsonl", [{
        "question": "A student has two groups of two objects. How many objects are there altogether?",
        "answer": "There are 2+2=4 objects altogether.\n#### 4",
    }])
    write_jsonl(root / "agents/DEEPSEEK_HARNESS_DOCS_AND_TOOLS.jsonl", [{
        "path": "README.md", "content": "Documentation for a safe agent harness. " * 8,
        "repository": "https://example.test/repo", "commit": "abc",
    }])
    (root / "cyber").mkdir(parents=True)
    (root / "cyber/CISA_KEV_current.json").write_text(json.dumps({
        "catalogVersion": "test", "vulnerabilities": [{
            "cveID": "CVE-2099-0001", "vendorProject": "Vendor", "product": "Product",
            "vulnerabilityName": "A vulnerability", "shortDescription": "A defensive description.",
            "requiredAction": "Apply the vendor update immediately.", "dateAdded": "2099-01-01",
            "dueDate": "2099-01-02",
        }],
    }), encoding="utf-8")
    aqua = root / "math/AQuA.zip"
    with zipfile.ZipFile(aqua, "w") as archive:
        train = {
            "question": "Choose the correct result when three objects are added to four other objects.",
            "options": ["A) 6", "B) 7"], "rationale": "Adding the two groups gives 3+4=7.", "correct": "B",
        }
        test = {"question": "THIS TEST MUST NEVER APPEAR", "correct": "A"}
        archive.writestr("train.json", json.dumps(train) + "\n" + json.dumps(train) + "\n")
        archive.writestr("test.json", json.dumps([test]))


def test_build_is_reproducible_and_never_creates_test(tmp_path):
    source = tmp_path / "source"
    fixture_sources(source)
    first, second = tmp_path / "first", tmp_path / "second"
    report1 = MODULE.build(source, first)
    report2 = MODULE.build(source, second)
    assert report1["test_created"] is False
    assert not (first / "test.jsonl").exists()
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    combined = (first / "train.jsonl").read_text() + (first / "validation.jsonl").read_text()
    assert "THIS TEST MUST NEVER APPEAR" not in combined
    assert set(report1["source_characters"]) == set(MODULE.SOURCE_POLICY)


def test_redacts_known_secret_shapes():
    fake_token = "hf_" + "A" * 32
    cleaned, findings = MODULE.redact_sensitive(f"token {fake_token} and a@b.com")
    assert "hf_" not in cleaned
    assert "a@b.com" not in cleaned
    assert findings == ["email", "huggingface_token"]


def test_group_split_is_deterministic():
    assert MODULE.stable_split("same-group") == MODULE.stable_split("same-group")
    assert MODULE.stable_split("same-group") in {"train", "validation"}


def test_proposed_training_mix_is_normalized_and_keeps_base_majority():
    config = json.loads((REPOSITORY_ROOT / "configs/corpus_mix_v10.json").read_text())
    assert abs(sum(config["weights"].values()) - 1.0) < 1e-12
    assert config["weights"]["ivoireslm_corpus_v0.9.0_base"] >= 0.6
    assert config["rules"]["never_sample_evaluation_data"] is True
    assert config["rules"]["test_remains_sealed"] is True
