def build_prompt_flan_large(user_query, retrieved_texts):
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)

    return f"""
You are an expert product selection assistant.

GOAL:
Choose exactly ONE product from the list that best matches the user's query.
Then extract all important information from that product.

REQUIRED OUTPUT FIELDS:
title:
category:
price:
list_price:
rating:
reviews:
best_seller:
bought_last_month:
product_url:
reason: (one short sentence)

RULES:
- Use ONLY data from the selected product.
- If a field is missing, leave it blank.
- No extra text outside the fields.

EXAMPLE:
Query: "budget earbuds"
Products:
Product A: Budget Earbuds | Electronics > Earbuds | $15 | Was $25 | 4.1 stars | 900 reviews | No | 1500 bought | https://amazon.com/a1
Product B: Sony Premium Headphones | Electronics > Headphones | $399 | Was $449 | 4.7 stars | 5000 reviews | Yes | 12000 bought | https://amazon.com/a2

Output:
title: Budget Earbuds
category: Electronics > Earbuds
price: $15
list_price: $25
rating: 4.1
reviews: 900
best_seller: No
bought_last_month: 1500
product_url: https://amazon.com/a1
reason: Best budget-friendly earbuds for the query.

REAL DATA:
Query: {user_query}
Products:
{retrieved_texts}

OUTPUT:
title:
"""

def build_prompt_flan_base(user_query, retrieved_texts):
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)

    return f"""
You must choose EXACTLY ONE product from the list that best matches the user query.

Your output MUST contain ALL of these fields in this exact order:

title:
category:
price:
list_price:
rating:
reviews:
best_seller:
bought_last_month:
product_url:
reason:

RULES:
- Use ONLY information from the selected product.
- Do NOT add any extra text before or after the fields.
- Do NOT explain.
- Do NOT output "No extra text".
- If a field has no data, leave it blank.

GOOD EXAMPLE:
title: Budget Earbuds
category: Electronics > Earbuds
price: $15
list_price: $25
rating: 4.1
reviews: 900
best_seller: No
bought_last_month: 1500
product_url: https://amazon.com/a1
reason: Best match for the budget-earbuds query.

BAD EXAMPLES (do NOT produce these):
- "No extra text."
- "The best product is..."
- "Output:"
- JSON format
- Any sentence outside the fields

USER QUERY:
{user_query}

PRODUCT LIST:
{retrieved_texts}

NOW PRODUCE THE OUTPUT BELOW.
title:
"""

def build_prompt_flan_small(user_query, retrieved_texts):
    if isinstance(retrieved_texts, list):
        retrieved_texts = "\n".join(retrieved_texts)

    return f"""
Choose ONE best product for the query.

Output these fields only:
title:
category:
price:
list_price:
rating:
reviews:
best_seller:
bought_last_month:
product_url:
reason:

Query: {user_query}
Products:
{retrieved_texts}

Output:
title:
"""
