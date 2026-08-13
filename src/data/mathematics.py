import re
import unicodedata

from bs4 import BeautifulSoup


SPACE_RE = re.compile(r"[ \t\u00a0]+")
BLANK_RE = re.compile(r"\n{3,}")
DISPLAYSTYLE_RE = re.compile(r"^\{\\displaystyle\s*(.*)\}$", re.DOTALL)
EXCLUDED_SECTIONS = {
    "annexes",
    "bibliographie",
    "liens externes",
    "notes",
    "notes et références",
    "références",
    "sources",
    "voir aussi",
}


def normalize_heading(value):
    return SPACE_RE.sub(" ", value).strip().casefold()


def clean_latex(value):
    latex = SPACE_RE.sub(" ", unicodedata.normalize("NFC", value)).strip()
    match = DISPLAYSTYLE_RE.match(latex)
    if match:
        latex = match.group(1).strip()
    return f"${latex}$" if latex else ""


def rendered_math_html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select(
        "script, style, figure, .mw-editsection, .reference, .references, "
        ".navbox, .noprint, .metadata, .sistersitebox, .mw-empty-elt"
    ):
        tag.decompose()

    for heading in list(soup.find_all(["h2", "h3"])):
        if normalize_heading(heading.get_text(" ", strip=True)) not in EXCLUDED_SECTIONS:
            continue
        level = heading.name
        sibling = heading.next_sibling
        while sibling is not None:
            following = sibling.next_sibling
            if getattr(sibling, "name", None) == level:
                break
            sibling.extract()
            sibling = following
        heading.decompose()

    for math in soup.find_all("math"):
        latex = math.get("alttext")
        if not latex:
            annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
            latex = annotation.get_text() if annotation else ""
        math.replace_with(clean_latex(latex))

    blocks = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd"]):
        if tag.find_parent(["table", "li"]):
            continue
        text = SPACE_RE.sub(" ", tag.get_text(" ", strip=True)).strip()
        if text and sum(character.isalpha() for character in text) >= 3:
            blocks.append(text)
    text = "\n".join(blocks)
    return BLANK_RE.sub("\n\n", unicodedata.normalize("NFC", text)).strip()
