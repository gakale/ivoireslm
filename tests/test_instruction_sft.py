from data.instruction_sft import core_rows, grounded_wdi_row, parse_wdi_sentence


def test_core_splits_are_disjoint_and_well_formed():
    splits = {name: core_rows(name) for name in ("train", "validation", "test")}
    prompts = {name: {row["prompt"] for row in rows} for name, rows in splits.items()}
    assert prompts["train"].isdisjoint(prompts["validation"])
    assert prompts["train"].isdisjoint(prompts["test"])
    assert prompts["validation"].isdisjoint(prompts["test"])
    assert all(row["target"].endswith("\n") for rows in splits.values() for row in rows)


def test_three_wdi_templates_are_parsed():
    samples = (
        "Selon les Indicateurs du développement dans le monde de la Banque mondiale, la valeur de l’indicateur « Test indicator » pour la Côte d’Ivoire en 1970 est 12,5.",
        "Pour la Côte d’Ivoire, la Banque mondiale rapporte une valeur de 12,5 en 1970 pour l’indicateur « Test indicator ».",
        "En 1970, l’indicateur « Test indicator » atteint 12,5 en Côte d’Ivoire dans la base WDI de la Banque mondiale.",
    )
    for index, sentence in enumerate(samples):
        parsed = parse_wdi_sentence(sentence)
        assert parsed == {"indicator": "Test indicator", "year": "1970", "value": "12,5"}
        row = grounded_wdi_row(sentence, index)
        assert row is not None
        assert row["target"] == " 12,5\n"
