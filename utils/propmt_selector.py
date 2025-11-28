from models.prompts import (
    build_prompt_cheapest,
    build_prompt_best_rated,
    build_prompt_most_purchased,
    build_prompt_feature_based,
    build_prompt_category_based,
    build_prompt_general,
)


def select_prompt(user_query, retrieved_texts):
    query_lower = user_query.lower()

    # Price-based
    if any(x in query_lower for x in ["cheapest", "under $", "lowest price", "affordable","price"]):
        return build_prompt_cheapest(user_query, retrieved_texts)

    # Rating-based
    elif any(x in query_lower for x in ["highest rating", "best rated", "top rated"]):
        return build_prompt_best_rated(user_query, retrieved_texts)

    # Popularity / purchased-based
    elif any(x in query_lower for x in ["bought frequently", "most purchased", "popular last month"]):
        return build_prompt_most_purchased(user_query, retrieved_texts)

    # Feature-based
    elif any(x in query_lower for x in ["durable", "sturdy", "lightweight", "travel-friendly"]):
        return build_prompt_feature_based(user_query, retrieved_texts)

    # Category-based
    elif any(x in query_lower for x in ["suitcase", "luggage", "bag", "men's clothing", "laptop"]):
        return build_prompt_category_based(user_query, retrieved_texts)

    # Default fallback
    else:
        return build_prompt_general(user_query, retrieved_texts)
