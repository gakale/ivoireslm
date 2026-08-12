def render_fish_meat_trade_sentence(
    year, category, subcategory, item_type, formatted_value
):
    context = (
        f"En {year}, dans la catégorie « {category} » et la "
        f"sous-catégorie « {subcategory} »"
    )
    if item_type:
        return f"{context}, le type « {item_type} » présente {formatted_value}."
    return f"{context}, la donnée présente {formatted_value}."
