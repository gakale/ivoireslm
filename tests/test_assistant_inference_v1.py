import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(relative: str, name: str):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


evaluation = load_script(
    "scripts/evaluation/evaluate_assistant_v1_17m.py", "evaluate_assistant_v1"
)
interface = load_script(
    "scripts/inference/gradio_assistant_v1_17m.py", "gradio_assistant_v1"
)


class FakeGeneration:
    text = "Je m’appelle IvoireSLM."
    stopped_on_eos = True


class FakeRuntime:
    model_id = "microivoire_transformer_v1.1_17m_assistant_pilot"

    def generate(self, _prompt):
        return FakeGeneration()

    def __call__(self, prompt):
        return f"modèle::{prompt}"


def test_normalization_handles_typography():
    assert evaluation.normalize(" Côte d’Ivoire ! ") == evaluation.normalize("côte d'ivoire")


def test_repetition_guard_detects_loop():
    assert evaluation.non_repetitive("une réponse courte et utile")
    assert not evaluation.non_repetitive("un deux trois un deux trois un deux trois un deux trois")


def test_model_only_mode_exposes_real_model_route():
    answer = interface.build_controller(FakeRuntime())
    response, route, status = answer("Comment t’appelles-tu ?", "Modèle seul")
    assert response == "Je m’appelle IvoireSLM."
    assert route == FakeRuntime.model_id
    assert "EOS" in status


def test_hybrid_mode_routes_exact_arithmetic():
    answer = interface.build_controller(FakeRuntime())
    response, route, status = answer("combien font 2 + 2 ?", "Assistant hybride")
    assert "Réponse : 4" in response
    assert route == "deterministic_math_tool_v0.1"
    assert "vérifiée" in status
