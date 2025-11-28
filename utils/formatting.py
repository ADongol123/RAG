def format_products_for_llm(df):
    lines = []
    for _, row in df.iterrows():
        line = (
            f"Title: {row['title']} | "
            f"Category: {row['category_name_x']} | "
            f"Price: {row['price']} | "
            f"Rating: {row['stars']} | "
            f"Reviews: {row['reviews']} | "
            f"BestSeller: {row['isBestSeller']} | "
            f"BoughtLastMonth: {row['boughtInLastMonth']}"
        )
        lines.append(line)
    return "\n".join(lines)


def format_products_for_llm_short(df):
    short_products = []
    for _, row in df.iterrows():
        short_line = f"{row['title']} (${row['price']}, {row['stars']}★, bought {row['boughtInLastMonth']})"
        short_products.append(short_line)
    return short_products
