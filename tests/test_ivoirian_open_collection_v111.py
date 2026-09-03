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


COLLECT = load(
    "snapshot_datagouvci_open_v011",
    "scripts/data/snapshot_datagouvci_open_v011.py",
)


def metadata(**overrides):
    result = {
        "id": "aggregate-education",
        "slug": "aggregate-education",
        "title": "Effectifs scolaires par région",
        "description": "Ce jeu présente des statistiques agrégées sur les écoles ivoiriennes.",
        "owner": {"name": "Ministère de l’Éducation"},
        "dataUpdatedAt": "2026-01-01T00:00:00Z",
        "license": {"title": "Licence Ouverte / Open Licence version 2.0"},
        "topics": [{"title": "Éducation"}],
        "schema": [
            {"key": "region", "x-originalName": "Région"},
            {"key": "effectif", "x-originalName": "Effectif"},
        ],
    }
    result.update(overrides)
    return result


def test_open_license_and_sensitive_schema_filters():
    assert COLLECT.open_license(metadata())
    assert not COLLECT.open_license(metadata(license={"title": "Inconnue"}))
    sensitive = metadata(schema=[{"key": "email", "x-originalName": "Courriel"}])
    assert COLLECT.sensitive_schema(sensitive) == ["email"]


def test_rendered_documents_are_grounded_and_group_split_is_stable():
    rows = [{"region": "Tonkpi", "effectif": 1234}, {"region": "Nawa", "effectif": 456}]
    documents = COLLECT.render_documents(metadata(), rows)
    assert len(documents) == 1
    document = documents[0]
    assert document["domain"] == "natural_ivoirian_grounded_verified"
    assert document["license"] == "Licence Ouverte / Open Licence"
    assert "Région : Tonkpi" in document["text"]
    assert document["split"] in {"train", "validation"}
    assert document["split"] == COLLECT.stable_split("aggregate-education")
