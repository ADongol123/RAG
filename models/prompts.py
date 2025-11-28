def build_prompt_cheapest(user_query, retrieved_texts):
    print("Building cheapest prompt")
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)
    return f"""
You are an expert product recommendation assistant. The user asked: "{user_query}". 
Select the **cheapest product** that satisfies the user's question from the list below and include all key details.

Products:
{retrieved_texts}

Instructions:
- Include: title, category, price, list price, rating, reviews, Best Seller status, bought last month, product URL.
- Write in 1-2 natural sentences explaining why this is the cheapest and suitable choice for the user.
- Do not list multiple products.
Answer:
"""


def build_prompt_best_rated(user_query, retrieved_texts):
    print("Building best rated prompt")
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)
    return f"""
You are an expert product recommendation assistant. The user asked: "{user_query}". 
Select the **highest rated product** from the list below and include all key details.

Products:
{retrieved_texts}

Instructions:
- Include: title, category, price, list price, rating, reviews, Best Seller status, bought last month, product URL.
- Write in 1-2 natural sentences explaining why this product is highly rated and relevant to the user's query.
- Do not list multiple products.
Answer:
"""


def build_prompt_most_purchased(user_query, retrieved_texts):
    print("Building most purchased prompt")
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)
    return f"""
You are an expert product recommendation assistant. The user asked: "{user_query}". 
Select the **most purchased product recently** from the list below and include all key details.

Products:
{retrieved_texts}

Instructions:
- Include: title, category, price, list price, rating, reviews, Best Seller status, bought last month, product URL.
- Write in 1-2 natural sentences explaining why this product is popular and suitable for the user's needs.
- Do not list multiple products.
Answer:
"""


def build_prompt_feature_based(user_query, retrieved_texts):
    print("Building feature based prompt")
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)
    return f"""
You are an expert product recommendation assistant. The user asked: "{user_query}". 
Select the product that **best matches the requested feature** (e.g., durable, lightweight, travel-friendly) 
and include all key details.

Products:
{retrieved_texts}

Instructions:
- Include: title, category, price, list price, rating, reviews, Best Seller status, bought last month, product URL.
- Write in 1-2 natural sentences explaining why this product matches the feature and is suitable for the user's request.
- Do not list multiple products.
Answer:
"""


def build_prompt_category_based(user_query, retrieved_texts):
    print("Building category based prompt")
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)
    return f"""
You are an expert product recommendation assistant. The user asked: "{user_query}". 
Select the most relevant product in the requested category and include all key details.

Products:
{retrieved_texts}

Instructions:
- Include: title, category, price, list price, rating, reviews, Best Seller status, bought last month, product URL.
- Write in 1-2 natural sentences explaining why this product fits the category and the user's request.
- Do not list multiple products.
Answer:
"""


def build_prompt_general(user_query, retrieved_texts):
    print("Building general prompt")
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)
    return f"""
You are an expert product recommendation assistant. The user asked: "{user_query}". 
Analyze the list carefully and select the single product that best answers the user's question.

Products:
{retrieved_texts}

Instructions:
- Include: title, category, price, list price, rating, reviews, Best Seller status, bought last month, product URL.
- Write in 1-2 natural sentences explaining why this product is the best choice for the user's request.
- Do not list multiple products.
Answer:
"""
