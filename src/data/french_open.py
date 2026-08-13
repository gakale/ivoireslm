import re
import unicodedata


ROLE_RE = re.compile(r":[a-zA-Z0-9_-]+:`([^`]+)`")
TICKS_RE = re.compile(r"`{1,2}([^`]+)`{1,2}")
DIRECTIVE_RE = re.compile(r"\.\.\s+[a-zA-Z0-9_-]+::\s*")
SPACE_RE = re.compile(r"\s+")


def clean_open_french_text(value):
    text = unicodedata.normalize("NFC", str(value))
    text = ROLE_RE.sub(lambda match: match.group(1).split(" <", 1)[0], text)
    text = TICKS_RE.sub(r"\1", text)
    text = DIRECTIVE_RE.sub("", text)
    text = text.replace("\\\n", " ")
    return SPACE_RE.sub(" ", text).strip()


def valid_documentation_block(text):
    if len(text) < 40 or len(text) > 20_000:
        return False
    return sum(character.isalpha() for character in text) >= 20


def dictionary_senses(entry):
    definitions = []
    seen = set()
    for sense in entry.get("senses", []):
        if sense.get("form_of") or sense.get("alt_of"):
            continue
        glosses = sense.get("glosses") or []
        if not glosses:
            continue
        definition = clean_open_french_text(glosses[-1])
        key = definition.casefold()
        if len(definition) >= 20 and key not in seen:
            seen.add(key)
            labels = []
            for label in sense.get("raw_tags") or sense.get("tags") or []:
                cleaned_label = clean_open_french_text(label)
                if cleaned_label and cleaned_label not in {"form-of", "alt-of"}:
                    labels.append(cleaned_label)
            definitions.append({"definition": definition, "labels": labels[:5]})
    return definitions


def dictionary_definitions(entry):
    return [sense["definition"] for sense in dictionary_senses(entry)]


def render_dictionary_entry(entry):
    word = clean_open_french_text(entry["word"])
    part_of_speech = clean_open_french_text(
        entry.get("pos_title") or entry.get("pos") or "catégorie non renseignée"
    )
    senses = dictionary_senses(entry)
    if not senses:
        raise ValueError("Une définition Wiktionnaire est requise")
    sections = [f"Dans le Wiktionnaire français, « {word} » est classé comme {part_of_speech}."]
    etymologies = entry.get("etymology_texts") or []
    if etymologies:
        etymology = clean_open_french_text(etymologies[0])
        if len(etymology) >= 10:
            sections.append(f"Étymologie indiquée : {etymology}")
    for index, sense in enumerate(senses, 1):
        usage = ""
        if sense["labels"]:
            usage = " (marques d’usage : " + ", ".join(sense["labels"]) + ")"
        sections.append(f"Sens {index}{usage} : {sense['definition']}")
    return " ".join(sections)
