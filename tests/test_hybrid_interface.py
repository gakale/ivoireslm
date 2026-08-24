import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "inference" / "gradio_hybrid_v01.py"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(SCRIPT.parent))

specification = importlib.util.spec_from_file_location("gradio_hybrid_v01", SCRIPT)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)


class FakeGenerator:
    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, prompt):
        return f"GÉNÉRATION::{prompt}"


def _controller(monkeypatch):
    monkeypatch.setattr(module, "TransformerGenerator", FakeGenerator)
    return module.build_controller(
        Path("checkpoint.pt"),
        Path("data"),
        Path("model.py"),
        max_new_tokens=100,
        temperature=0.7,
        top_k=30,
        seed=1,
    )


def test_interface_exposes_verified_math_route(monkeypatch):
    answer = _controller(monkeypatch)
    response, route, status, detail = answer("Calculer 18 + 24.")
    assert "Réponse : 42" in response
    assert route == "deterministic_math_tool_v0.1"
    assert "vérifiée" in status
    assert detail == "Calcul déterministe contrôlé."


def test_interface_exposes_unverified_transformer_route(monkeypatch):
    answer = _controller(monkeypatch)
    response, route, status, detail = answer("La Côte d’Ivoire")
    assert response == "GÉNÉRATION::La Côte d’Ivoire"
    assert route == "microivoire_transformer_v0.2_5m"
    assert "non vérifiée" in status
    assert "vérifier" in detail


def test_interface_handles_empty_input(monkeypatch):
    answer = _controller(monkeypatch)
    response, route, status, _detail = answer("  ")
    assert "question" in response
    assert route == "aucune"
    assert status == "—"
