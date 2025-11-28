from models.embedder import search_product
from models.llm_models import model, tokenizer, model2, tokenizer2, model3, tokenizer3, run_model
from utils.formatting import format_products_for_llm
from utils.propmt_selector import select_prompt
def rag_pipeline_all_models(user_query, top_k=5):
    # Retrieve products
    retrieved_df = search_product(user_query, k=top_k)
    retrieved_full = format_products_for_llm(retrieved_df)

    # Select prompts dynamically
    prompt_normal = select_prompt(user_query, retrieved_full)
    print(prompt_normal)
    # Run models
    result = {
        "normal": {
            "flan_large": run_model(model, tokenizer, prompt_normal),
            "flan_base": run_model(model2, tokenizer2, prompt_normal),
            "flan_small": run_model(model3, tokenizer3, prompt_normal),
        },
    }

    return result


def rag_pipeline_hf(user_query, top_k=10):
    return rag_pipeline_all_models(user_query, top_k)
