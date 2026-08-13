import decimal


def format_faostat_value(value):
    """Render an exact FAOSTAT decimal with French separators."""
    try:
        number = decimal.Decimal(str(value).strip())
    except decimal.InvalidOperation as error:
        raise ValueError("La valeur FAOSTAT doit être numérique") from error
    if not number.is_finite():
        raise ValueError("La valeur FAOSTAT doit être finie")
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    integer, dot, fraction = rendered.partition(".")
    sign = ""
    if integer.startswith("-"):
        sign, integer = "-", integer[1:]
    groups = []
    while integer:
        groups.append(integer[-3:])
        integer = integer[:-3]
    result = sign + "\u202f".join(reversed(groups))
    if dot and fraction:
        result += "," + fraction
    return result


def clean_code(value):
    return " ".join(str(value).strip().lstrip("'").split())


def render_faostat_production_sentence(row, template_index=0):
    values = {
        "year": int(row["year"]),
        "item": " ".join(str(row["item"]).split()),
        "item_code": clean_code(row["item_code"]),
        "cpc_code": clean_code(row["item_code_cpcx"]),
        "element": " ".join(str(row["element"]).split()),
        "element_code": clean_code(row["element_code"]),
        "value": format_faostat_value(row["value"]),
        "unit": " ".join(str(row["unit"]).split()),
        "flag": clean_code(row["flag"]),
    }
    templates = (
        "Selon FAOSTAT, pour la Côte d’Ivoire en {year}, le produit officiel « {item} » "
        "(code FAOSTAT {item_code}; code CPC {cpc_code}) présente une valeur de {value} "
        "{unit} pour l’élément « {element} » (code {element_code}), avec le drapeau de "
        "qualité FAO « {flag} ».",
        "Dans le domaine FAOSTAT Production: Crops and livestock products, l’observation "
        "ivoirienne de {year} associe « {item} » (FAOSTAT {item_code}; CPC {cpc_code}) à "
        "l’élément « {element} » (code {element_code}) : {value} {unit}; drapeau FAO "
        "« {flag} ».",
        "FAOSTAT rapporte pour la Côte d’Ivoire, en {year}, {value} {unit} pour « {element} » "
        "(code {element_code}) concernant « {item} » (code FAOSTAT {item_code}; code CPC "
        "{cpc_code}); le drapeau de qualité est « {flag} ».",
    )
    return templates[template_index % len(templates)].format(**values)
