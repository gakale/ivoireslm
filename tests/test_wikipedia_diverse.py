import importlib.util
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/data/snapshot_wikipedia_fr_diverse_v01.py"
SPEC = importlib.util.spec_from_file_location("snapshot_wikipedia_diverse", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fetch_page_keeps_revision_attribution_and_full_extract(tmp_path, monkeypatch):
    page_id = 987654
    text = "Un article naturel et suffisamment long. " * 30

    def fake_request(parameters):
        assert parameters["pageids"] == str(page_id)
        assert parameters["exlimit"] == "max"
        return {
            "query": {
                "pages": [
                    {
                        "pageid": page_id,
                        "title": "Article ivoirien de test",
                        "extract": text,
                        "lastrevid": 123,
                        "touched": "2026-08-29T00:00:00Z",
                        "fullurl": "https://fr.wikipedia.org/wiki/Test",
                    }
                ]
            }
        }

    monkeypatch.setattr(MODULE, "api_request", fake_request)
    discovered = {
        "pageid": page_id,
        "title": "Article ivoirien de test",
        "category_labels": ["cote_ivoire"],
        "root_categories": ["Catégorie:Côte d'Ivoire"],
    }
    MODULE.fetch_page(tmp_path, discovered, previous_page_ids={page_id})
    cached = json.loads(MODULE.cache_path(tmp_path, page_id).read_text(encoding="utf-8"))
    assert cached["revision_id"] == 123
    assert cached["split"] == "train"
    assert cached["seen_in_previous_snapshot"] is True
    assert len(cached["text"]) >= 500
