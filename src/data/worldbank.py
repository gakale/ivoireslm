import math


def format_worldbank_value(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("La valeur Banque mondiale doit être finie")
    if number.is_integer() and abs(number) <= 9_007_199_254_740_992:
        rendered = str(int(number))
    else:
        rendered = format(number, ".12g")
    if "e" in rendered.lower():
        mantissa, exponent = rendered.lower().split("e")
        rendered = f"{mantissa.replace('.', ',')} × 10^{int(exponent)}"
        return rendered
    integer, dot, fraction = rendered.partition(".")
    sign = ""
    if integer.startswith("-"):
        sign, integer = "-", integer[1:]
    groups = []
    while integer:
        groups.append(integer[-3:])
        integer = integer[:-3]
    rendered = sign + "\u202f".join(reversed(groups))
    if dot:
        rendered += "," + fraction
    return rendered


def render_wdi_sentence(year, indicator_name, value, template_index=0):
    formatted = format_worldbank_value(value)
    templates = (
        "Selon les Indicateurs du développement dans le monde de la Banque mondiale, "
        "la valeur de l’indicateur « {indicator} » pour la Côte d’Ivoire en {year} "
        "est {value}.",
        "Pour la Côte d’Ivoire, la Banque mondiale rapporte une valeur de {value} en "
        "{year} pour l’indicateur « {indicator} ».",
        "En {year}, l’indicateur « {indicator} » atteint {value} en Côte d’Ivoire "
        "dans la base WDI de la Banque mondiale.",
    )
    return templates[template_index % len(templates)].format(
        year=year, indicator=indicator_name, value=formatted
    )
