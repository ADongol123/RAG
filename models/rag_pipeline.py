from models.embedder import search_product
from models.llm_models import model, tokenizer, model2, tokenizer2, model3, tokenizer3, run_model
from utils.formatting import format_products_for_llm
# from utils.propmt_selector import select_prompt
from models.prompts import build_prompt_flan_large, build_prompt_flan_base, build_prompt_flan_small
def rag_pipeline_all_models(user_query, top_k=5):
    # Retrieve products
    retrieved_df = search_product(user_query, k=top_k)
    retrieved_full = format_products_for_llm(retrieved_df)
    print("Retrieved Products:", retrieved_full)
    # Select prompts dynamically
    # prompt_normal = select_prompt(user_query, retrieved_full)
    prompt_large = build_prompt_flan_large(user_query, retrieved_full)
    prompt_base = build_prompt_flan_base(user_query, retrieved_full)
    prompt_small = build_prompt_flan_small(user_query, retrieved_full)
    # print(prompt_normal)
    # Run models
    result = {
        "normal": {
            "flan_large": run_model(model, tokenizer, prompt_large),
            "flan_base": run_model(model2, tokenizer2, prompt_base),
            "flan_small": run_model(model3, tokenizer3, prompt_small),
        },
    }

    return result


def rag_pipeline_hf(user_query, top_k=5):
    return rag_pipeline_all_models(user_query, top_k)
