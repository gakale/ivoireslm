"""Assistant outillé IvoireSLM 17M avec routes explicites et traçables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
import re
import unicodedata
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from tools.math_solver import UnsupportedMathProblem, solve_math_problem


@dataclass(frozen=True)
class AssistantResponse:
    response: str
    route: str
    status: str
    verified: bool
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class WebResult:
    title: str
    extract: str
    url: str


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold().replace("’", "'"))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9ɔɛ' +*/×÷-]+", " ", text).strip()


class WikipediaFrenchSearch:
    """Recherche en lecture seule via l'API Action officielle de MediaWiki."""

    endpoint = "https://fr.wikipedia.org/w/api.php"

    def __init__(self, timeout: float = 8.0, limit: int = 3):
        self.timeout = timeout
        self.limit = limit

    def search(self, query: str) -> list[WebResult]:
        query = " ".join(query.split())[:300]
        if len(query) < 2:
            return []
        parameters = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "0",
            "gsrlimit": str(self.limit),
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "3",
            "inprop": "url",
            "format": "json",
            "formatversion": "2",
        }
        request = Request(
            f"{self.endpoint}?{urlencode(parameters)}",
            headers={
                "User-Agent": (
                    "IvoireSLM-Research/1.0 "
                    "(educational language-model project; contact via project repository)"
                )
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"réponse HTTP {response.status}")
                payload = json.loads(response.read(1_000_000).decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"recherche Internet indisponible : {exc}") from exc

        pages = payload.get("query", {}).get("pages", [])
        results = []
        for page in sorted(pages, key=lambda item: item.get("index", 10_000)):
            title = " ".join(str(page.get("title", "")).split())
            extract = " ".join(html.unescape(str(page.get("extract", ""))).split())
            url = page.get("fullurl") or f"https://fr.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            if title and extract and str(url).startswith("https://fr.wikipedia.org/"):
                results.append(WebResult(title, extract[:900], str(url)))
        return results


IDENTITY_PATTERNS = (
    r"\b(?:comment (?:tu t'appelles|t'appelles[- ]tu)|quel est ton nom)\b",
    r"\b(?:tu t'appel(?:les|le)? comment|c'est quoi ton nom)\b",
    r"\bqui es[- ]tu\b",
    r"\b(?:tu (?:es|est) qui|t'es qui)\b",
    r"\bpresente[- ]toi\b",
    r"\bes[- ]tu (?:une personne|un etre humain|humain)\b",
    r"\bquel est ton role\b",
)

CONVERSATION_RULES = (
    (r"^(?:bonjour|bjr|coucou)\b", "Bonjour ! Comment puis-je t’aider ?"),
    (r"^(?:bonsoir)\b", "Bonsoir ! Comment puis-je t’aider ?"),
    (r"^(?:salut|slt)\b", "Salut ! Comment puis-je t’aider ?"),
    (r"\b(?:merci|je te remercie)\b", "Avec plaisir !"),
    (r"\b(?:au revoir|a plus tard|a demain)\b", "D’accord, à bientôt !"),
    (r"\b(?:comment vas tu|ca va)\b", "Je vais bien, merci. Et toi ?"),
    (r"\bje vais bien\b", "Content de l’apprendre !"),
    (
        r"\b(?:tu fais quoi|que fais[- ]tu) (?:aujourd'hui|maintenant)\b",
        "Je suis disponible pour répondre à tes questions et utiliser mes outils.",
    ),
    (
        r"\b(?:tu sais|sais[- ]tu|tu peux|peux[- ]tu|tu sais faire) (?:faire )?quoi\b",
        "Je peux effectuer certains calculs, donner des informations ivoiriennes vérifiées, répondre sur mon identité, utiliser un petit lexique dioula et rechercher des informations sur Internet si tu l’autorises.",
    ),
    (
        r"\b(?:tu dis quoi|j'ai pas compris|je n'ai pas compris|repete|répète)\b",
        "Dis-moi quelle réponse ou quelle partie tu n’as pas comprise, et je vais la reformuler plus simplement.",
    ),
)

IVOIRE_SOURCES = {
    "diplomatie": "https://diplomatie.gouv.ci/informations-utiles/presentation-de-la-c%C3%B4te-d-ivoire",
    "oif": "https://oif.diplomatie.gouv.ci/fiche_signaletique.php",
    "bceao": "https://www.bceao.int/fr/content/histoire-du-franc-cfa",
}


def _identity(text: str) -> AssistantResponse | None:
    if not any(re.search(pattern, text) for pattern in IDENTITY_PATTERNS):
        return None
    if re.search(r"\b(?:personne|etre humain|humain)\b", text):
        answer = "Non. Je suis IvoireSLM, un petit modèle de langage expérimental, pas une personne."
    elif "role" in text:
        answer = "Je suis IvoireSLM. Mon rôle est d’aider à répondre aux questions avec mes outils et mes connaissances limitées."
    else:
        answer = "Je suis IvoireSLM, un petit modèle de langage expérimental centré sur la Côte d’Ivoire."
    return AssistantResponse(answer, "identity_card_v1", "✅ identité contrôlée", True)


def _conversation(text: str) -> AssistantResponse | None:
    for pattern, answer in CONVERSATION_RULES:
        if re.search(pattern, text):
            return AssistantResponse(answer, "conversation_rules_v1", "✅ réponse conversationnelle contrôlée", True)
    return None


def _dioula(text: str) -> AssistantResponse | None:
    if not re.search(r"\b(?:dioula|jula)\b", text):
        return None
    if re.search(r"\b(?:bonjour|salutation|matin)\b", text):
        answer = "Une salutation courante du matin en dioula est « i ni sɔgɔma », aussi écrite « ani sogoma » selon les usages."
    elif re.search(r"\bmerci\b", text):
        answer = "On peut dire « i ni ce » pour remercier quelqu’un en dioula."
    else:
        answer = "Le dioula, aussi appelé jula, est une langue mandingue parlée notamment en Côte d’Ivoire et au Burkina Faso."
    return AssistantResponse(
        answer,
        "verified_dioula_lexicon_v1",
        "✅ entrée du petit lexique vérifié",
        True,
        ("Koumankan4Dyula — validation humaine requise pour toute extension",),
    )


def _ivoire_fact(text: str) -> AssistantResponse | None:
    mentions_country = bool(re.search(r"\b(?:cote d'ivoire|ivoirien|ivoirienne|le pays)\b", text))
    if not mentions_country:
        return None
    if re.search(r"\bcapital(?:e)? economique\b", text):
        return AssistantResponse(
            "Abidjan est la capitale économique et la principale ville de Côte d’Ivoire.",
            "verified_ivoire_kb_v1", "✅ fait institutionnel vérifié", True,
            (IVOIRE_SOURCES["diplomatie"],),
        )
    if re.search(r"\bcapital(?:e)?\b", text):
        return AssistantResponse(
            "La capitale politique et administrative de la Côte d’Ivoire est Yamoussoukro. Abidjan est sa capitale économique.",
            "verified_ivoire_kb_v1", "✅ fait institutionnel vérifié", True,
            (IVOIRE_SOURCES["diplomatie"],),
        )
    facts = (
        (("langue officielle",), "Le français est la langue officielle de la Côte d’Ivoire.", "diplomatie"),
        (("monnaie", "devise monetaire"), "La Côte d’Ivoire utilise le franc CFA de l’UEMOA, dont le code est XOF.", "bceao"),
        (("independance",), "La Côte d’Ivoire a proclamé son indépendance le 7 août 1960.", "oif"),
        (("fete nationale",), "La fête nationale ivoirienne est célébrée le 7 août.", "diplomatie"),
        (("devise nationale", "union discipline travail"), "La devise nationale est « Union – Discipline – Travail ».", "oif"),
        (("hymne",), "L’hymne national de la Côte d’Ivoire s’appelle L’Abidjanaise.", "oif"),
        (("drapeau",), "Le drapeau ivoirien comporte trois bandes verticales orange, blanche et verte.", "oif"),
        (("ouest", "region d'afrique", "partie de l'afrique"), "La Côte d’Ivoire se situe en Afrique de l’Ouest.", "diplomatie"),
    )
    for keywords, answer, source in facts:
        if any(keyword in text for keyword in keywords):
            return AssistantResponse(
                answer, "verified_ivoire_kb_v1", "✅ fait institutionnel vérifié", True,
                (IVOIRE_SOURCES[source],),
            )
    return None


def _local_time(text: str) -> AssistantResponse | None:
    asks_time = bool(re.search(
        r"\b(?:quelle heure|quel heure|qu'elle heure|heure est[- ]il|il est (?:quelle|quel|qu'elle) heure)\b",
        text,
    ))
    asks_abidjan = bool(re.search(r"\b(?:abidjan|cote d'ivoire|ivoirienne?)\b", text))
    if not (asks_time and asks_abidjan):
        return None
    current = datetime.now(ZoneInfo("Africa/Abidjan"))
    return AssistantResponse(
        f"À Abidjan, il est actuellement {current:%H:%M} (heure locale, UTC+0).",
        "local_time_tool_v1",
        "✅ heure calculée au moment de la requête",
        True,
    )


def _internet_requested(text: str) -> bool:
    return bool(re.search(r"\b(?:cherche|recherche|verifie) (?:sur )?(?:internet|le web|wikipedia|en ligne)\b", text))


def _strip_internet_command(request: str) -> str:
    cleaned = re.sub(
        r"\b(?:cherche|recherche|vérifie|verifie) (?:sur )?(?:internet|le web|wikipedia|en ligne)\b[: ]*",
        "", request, flags=re.IGNORECASE,
    ).strip(" :,-")
    return cleaned or request.strip()


def _looks_factual(text: str) -> bool:
    return bool(re.match(
        r"^(?:qui|que|quoi|quel|quelle|quels|quelles|quand|ou|pourquoi|comment|"
        r"c'est quoi|c'est qui|qu'est ce|definis|explique|tu connais|"
        r"il est (?:quelle|quel|qu'elle) heure)\b",
        text,
    ))


def _trim_repetition(response: str) -> tuple[str, bool]:
    """Interrompt une boucle évidente sans prétendre corriger le fond."""
    text = " ".join(response.split()).strip()
    marker = re.search(r"\bAssistant\s*:", text, flags=re.IGNORECASE)
    if marker:
        text = text[: marker.start()].strip()
        return text, True
    words = text.split()
    for width in range(3, min(13, len(words) // 2 + 1)):
        for start in range(0, len(words) - 2 * width + 1):
            first = [normalize(word) for word in words[start : start + width]]
            second = [normalize(word) for word in words[start + width : start + 2 * width]]
            if first == second:
                return " ".join(words[: start + width]).strip(), True
    if len(text) > 500:
        return text[:500].rsplit(" ", 1)[0] + "…", True
    return text, False


def _web_answer(query: str, search: WikipediaFrenchSearch) -> AssistantResponse:
    try:
        results = search.search(query)
    except RuntimeError as exc:
        return AssistantResponse(
            "La recherche Internet est momentanément indisponible. Je préfère ne pas inventer de réponse.",
            "internet_search_error", f"⚠️ {exc}", False,
        )
    if not results:
        return AssistantResponse(
            "Je n’ai trouvé aucun résultat suffisamment clair dans cette recherche. Essaie de reformuler la question.",
            "wikipedia_search_v1", "⚠️ aucun résultat Internet", False,
        )
    primary = results[0]
    response = f"**{primary.title}** — {primary.extract}"
    return AssistantResponse(
        response,
        "wikipedia_search_v1",
        "🌐 extrait Internet à vérifier dans les sources",
        False,
        tuple(result.url for result in results),
    )


def route_assistant(
    request: str,
    language_generator: Callable[[str], str],
    *,
    internet_enabled: bool = False,
    web_search: WikipediaFrenchSearch | None = None,
) -> AssistantResponse:
    """Sélectionne une route déterministe avant le modèle ou Internet."""
    if not isinstance(request, str) or not request.strip():
        return AssistantResponse("Écris d’abord une question.", "none", "—", True)
    original = request.strip()
    text = normalize(original)

    # Les domaines explicites passent avant la conversation : « merci en
    # dioula » est une traduction, pas un remerciement adressé à l'assistant.
    for route in (_identity, _local_time, _dioula, _ivoire_fact, _conversation):
        result = route(text)
        if result is not None:
            return result

    # Les titulaires de fonctions publiques peuvent changer. On ne conserve
    # donc pas leur nom dans une règle statique.
    if re.search(r"\bpresident\b", text) and re.search(r"\b(?:cote d'ivoire|ivoirien)\b", text):
        if internet_enabled or _internet_requested(text):
            return _web_answer(_strip_internet_command(original), web_search or WikipediaFrenchSearch())
        return AssistantResponse(
            "Cette information peut changer. Active la recherche Internet pour que je consulte une source actuelle.",
            "freshness_guard_v1",
            "⚠️ recherche Internet nécessaire",
            True,
        )

    try:
        solution = solve_math_problem(original)
        return AssistantResponse(
            solution.completion.strip(),
            "deterministic_math_tool_v0.1",
            "✅ calcul déterministe vérifié",
            True,
        )
    except UnsupportedMathProblem:
        if re.search(r"\d\s*[+*/×÷-]\s*\d|\b(?:calcul|equation|fraction|pourcentage)\b", text):
            return AssistantResponse(
                "Je reconnais une question mathématique, mais mon calculateur ne prend pas encore cette forme en charge.",
                "unsupported_math_guard", "⚠️ calcul non pris en charge", True,
            )

    explicit_web = _internet_requested(text)
    if explicit_web or (internet_enabled and _looks_factual(text)):
        return _web_answer(_strip_internet_command(original), web_search or WikipediaFrenchSearch())

    if _looks_factual(text):
        return AssistantResponse(
            "Cette question demande une information que mes outils locaux ne contiennent pas. Active la recherche Internet pour obtenir une réponse accompagnée de sources.",
            "factual_freshness_guard_v1",
            "⚠️ recherche Internet désactivée",
            True,
        )

    response = language_generator(original).strip()
    if not response:
        response = "Je n’ai pas réussi à produire une réponse. Essaie de reformuler la question."
    response, repetition_stopped = _trim_repetition(response)
    return AssistantResponse(
        response,
        getattr(language_generator, "model_id", "microivoire_transformer_17m"),
        (
            "⚠️ répétition du modèle interrompue ; contenu non vérifié"
            if repetition_stopped
            else "⚠️ génération libre du modèle, non vérifiée"
        ),
        False,
    )
