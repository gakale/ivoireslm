import hashlib
import re
import unicodedata


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORD_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿŒœ’'-]+", re.UNICODE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"(?<![\d,.])(?:\+225[ .-]?(?:0[157][ .-]?)(?:\d[ .-]?){8}"
    r"|0[157](?:[ .-]\d{2}){4})(?!\d)"
)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = CONTROL_RE.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = []
    blank = False
    for line in lines:
        if not line.strip():
            if cleaned and not blank:
                cleaned.append("")
            blank = True
            continue
        cleaned.append(line.strip())
        blank = False
    return "\n".join(cleaned).strip() + "\n"


def normalized_line_key(line):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", line)).strip().casefold()


def word_count(text):
    return len(WORD_RE.findall(text))


def token_shingles(text, width=5):
    tokens = [token.casefold() for token in WORD_RE.findall(text)]
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0
