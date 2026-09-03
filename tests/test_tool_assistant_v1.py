import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.tool_assistant_v1 import WebResult, _prepare_web_query, route_assistant


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


def test_identity_accepts_natural_variant():
    for question in ("Tu es qui ?", "Tu est qui ?", "Tu t'appel comment ?"):
        result = route_assistant(question, FakeGenerator())
        assert result.route == "identity_card_v1"


def test_math_is_exact():
    result = route_assistant("Combien font 17 × 8 ?", FakeGenerator())
    assert result.route == "deterministic_math_tool_v0.1"
    assert "136" in result.response
    assert result.verified


def test_math_accepts_user_interface_label():
    result = route_assistant("Question : Combien font 17 × 8 ?", FakeGenerator())
    assert result.route == "deterministic_math_tool_v0.1"
    assert "136" in result.response


def test_math_verifies_a_claimed_result():
    correct = route_assistant("Calcul vérifié : 17 × 8 = 136", FakeGenerator())
    incorrect = route_assistant("Vérifie le calcul : 17 × 8 = 120", FakeGenerator())
    assert correct.route == "deterministic_math_tool_v0.1"
    assert "égalité correcte" in correct.response
    assert "résultat attendu : 136" in incorrect.response


def test_political_and_economic_capitals_are_distinguished():
    political = route_assistant("Quelle est la capitale de la Côte d’Ivoire ?", FakeGenerator())
    economic = route_assistant("Quelle est la capitale économique ivoirienne ?", FakeGenerator())
    assert "Yamoussoukro" in political.response
    assert "Abidjan" in economic.response
    assert political.sources and economic.sources


def test_capital_typo_is_tolerated():
    result = route_assistant("Qu'elle est la capital de la Côte d'Ivoire ?", FakeGenerator())
    assert result.route == "verified_ivoire_kb_v1"
    assert "Yamoussoukro" in result.response


def test_dioula_uses_small_verified_lexicon():
    result = route_assistant("Comment dit-on merci en dioula ?", FakeGenerator())
    assert result.route == "verified_dioula_lexicon_v1"
    assert "i ni ce" in result.response


def test_unknown_dioula_translation_is_not_invented():
    result = route_assistant("lavé en dioula", FakeGenerator())
    assert result.route == "dioula_lexicon_miss_v1"
    assert "ne contient pas" in result.response


def test_internet_requires_explicit_request_or_checkbox():
    raw = route_assistant("Qui est Marie Curie ?", FakeGenerator(), web_search=FakeSearch())
    web = route_assistant(
        "Qui est Marie Curie ?", FakeGenerator(), internet_enabled=True, web_search=FakeSearch()
    )
    assert raw.route == "factual_freshness_guard_v1"
    assert web.route == "wikipedia_search_v1"
    assert web.sources == ("https://fr.wikipedia.org/wiki/Test",)
    assert not web.verified


def test_unknown_definition_is_not_sent_to_unreliable_model():
    guarded = route_assistant("C'est quoi une maison ?", FakeGenerator())
    searched = route_assistant(
        "C'est quoi une maison ?", FakeGenerator(), internet_enabled=True,
        web_search=FakeSearch(),
    )
    assert guarded.route == "factual_freshness_guard_v1"
    assert searched.route == "wikipedia_search_v1"


def test_conversation_variants_are_controlled():
    capabilities = route_assistant("Tu sais faire quoi ?", FakeGenerator())
    clarification = route_assistant("Tu dis quoi, j'ai pas compris", FakeGenerator())
    assert capabilities.route == "conversation_rules_v1"
    assert clarification.route == "conversation_rules_v1"


def test_explicit_web_request_works_without_checkbox():
    result = route_assistant(
        "Cherche sur Internet qui est Marie Curie.", FakeGenerator(), web_search=FakeSearch()
    )
    assert result.route == "wikipedia_search_v1"


def test_current_president_requires_fresh_information():
    guarded = route_assistant("C'est qui le président de la Côte d'Ivoire ?", FakeGenerator())
    assert guarded.route == "freshness_guard_v1"


def test_web_query_removes_question_scaffolding():
    assert _prepare_web_query("C'est quoi une maison ?") == "maison"
    assert _prepare_web_query("Qui est Laurent Gbagbo ?") == "Laurent Gbagbo"


def test_slm_has_an_explicit_local_definition():
    result = route_assistant("slm", FakeGenerator(), internet_enabled=True)
    assert result.route == "local_ai_glossary_v1"
    assert "Small Language Model" in result.response


def test_abidjan_time_uses_clock_tool():
    for question in ("Il est quelle heure à Abidjan ?", "Il est qu'elle heure Abidjan ?"):
        result = route_assistant(question, FakeGenerator())
        assert result.route == "local_time_tool_v1"
        assert "UTC+0" in result.response
        assert result.verified


def test_assistant_fallback_stops_obvious_loop():
    class RepeatingModel(FakeGenerator):
        def __call__(self, _prompt):
            return "une même réponse utile une même réponse utile une même réponse utile"

    result = route_assistant("Raconte quelque chose.", RepeatingModel())
    assert result.response == "une même réponse utile"
    assert "interrompue" in result.status


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
