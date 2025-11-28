# from models.prompts import build_prompt

# def select_prompt(user_query, retrieved_texts):
#     """
#     Automatically selects the correct selection_mode based on user query
#     and builds a structured prompt for FLAN-T5 models.
#     """
#     query_lower = user_query.lower()

#     # 1️⃣ Determine selection mode
#     if any(x in query_lower for x in ["cheap", "cheapest", "under $", "lowest", "affordable", "price"]):
#         selection_mode = "cheapest"

#     elif any(x in query_lower for x in ["highest rating", "best rated", "top rated", "highest rated"]):
#         selection_mode = "best_rated"

#     elif any(x in query_lower for x in ["most purchased", "popular", "bought frequently", "frequently bought", "best seller"]):
#         selection_mode = "most_purchased"

#     elif any(x in query_lower for x in ["durable", "sturdy", "lightweight", "travel-friendly", "compact", "expandable"]):
#         selection_mode = "feature_based"

#     else:
#         selection_mode = "general"

#     # 2️⃣ Build structured prompt
#     return build_prompt(user_query, retrieved_texts, selection_mode)
