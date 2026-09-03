import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.tool_assistant_v1 import WebResult, route_assistant


class FakeGenerator:
    model_id = "fake_17m"

    def __call__(self, prompt):
        return f"LIBRE::{prompt}"


class FakeSearch:
    def search(self, query):
        return [WebResult("Résultat", f"Extrait pour {query}", "https://fr.wikipedia.org/wiki/Test")]


def test_identity_is_controlled():
    result = route_assistant("Comment t'appelles-tu ?", FakeGenerator())
    assert result.route == "identity_card_v1"
    assert result.verified
    assert "IvoireSLM" in result.response


def test_math_is_exact():
    result = route_assistant("Combien font 17 × 8 ?", FakeGenerator())
    assert result.route == "deterministic_math_tool_v0.1"
    assert "136" in result.response
    assert result.verified


def test_political_and_economic_capitals_are_distinguished():
    political = route_assistant("Quelle est la capitale de la Côte d’Ivoire ?", FakeGenerator())
    economic = route_assistant("Quelle est la capitale économique ivoirienne ?", FakeGenerator())
    assert "Yamoussoukro" in political.response
    assert "Abidjan" in economic.response
    assert political.sources and economic.sources


def test_dioula_uses_small_verified_lexicon():
    result = route_assistant("Comment dit-on merci en dioula ?", FakeGenerator())
    assert result.route == "verified_dioula_lexicon_v1"
    assert "i ni ce" in result.response


def test_internet_requires_explicit_request_or_checkbox():
    raw = route_assistant("Qui est Marie Curie ?", FakeGenerator(), web_search=FakeSearch())
    web = route_assistant(
        "Qui est Marie Curie ?", FakeGenerator(), internet_enabled=True, web_search=FakeSearch()
    )
    assert raw.route == "fake_17m"
    assert web.route == "wikipedia_search_v1"
    assert web.sources == ("https://fr.wikipedia.org/wiki/Test",)
    assert not web.verified


def test_explicit_web_request_works_without_checkbox():
    result = route_assistant(
        "Cherche sur Internet qui est Marie Curie.", FakeGenerator(), web_search=FakeSearch()
    )
    assert result.route == "wikipedia_search_v1"


def test_gradio_controller_keeps_model_only_mode_honest():
    script = ROOT / "scripts/inference/gradio_tool_assistant_v1.py"
    spec = importlib.util.spec_from_file_location("gradio_tool_assistant_v1", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Generation:
        text = "réponse brute"
        stopped_on_eos = True

    class Runtime(FakeGenerator):
        checkpoint_step = 500

        def generate(self, _prompt):
            return Generation()

    answer = module.build_controller(Runtime(), FakeSearch())
    response, route, status, sources, interaction = answer(
        "Qui es-tu ?", "Modèle seul", True
    )
    assert response == "réponse brute"
    assert route == "fake_17m"
    assert sources == ""
    assert interaction["internet_enabled"] is True
