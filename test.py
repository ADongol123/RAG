import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import nltk
import json
from models.rag_pipeline import rag_pipeline_all_models

nltk.download('punkt')

# =====================================================================
# SCORERS
# =====================================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# =====================================================================
# METRIC FUNCTIONS
# =====================================================================
def compute_rouge(pred, reference):
    score = rouge.score(reference, pred)
    return score["rougeL"].fmeasure

def retrieval_similarity(query_emb, retrieved_emb):
    sims = util.cos_sim(query_emb, retrieved_emb)
    return float(sims.mean())

def grounding_rate(pred, retrieved_text):
    sentences = nltk.sent_tokenize(pred)
    retrieved_emb = embedder.encode(retrieved_text, convert_to_tensor=True)
    grounded = 0
    for s in sentences:
        s_emb = embedder.encode(s, convert_to_tensor=True)
        if util.cos_sim(s_emb, retrieved_emb).max().item() > 0.55:
            grounded += 1
    return grounded / len(sentences) if sentences else 0

def fluency_score(text):
    sentences = nltk.sent_tokenize(text)
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) <= 1:
        return 1.0
    var = np.var(lengths)
    return float(1 / (1 + var))

# =====================================================================
# YOUR QUERIES & REFERENCES
# =====================================================================
queries = [
    "Show me the cheapest suitcase under $100",
    "Find me a lightweight suitcase with good reviews",
    "Which suitcase has the highest rating?",
    "Recommend a durable suitcase for travel",
    "Show me suitcases bought frequently last month",
    "List the best seller suitcase options"
]

references = [
    "The cheapest suitcase under $100 is the YESSUIT Travel Spinner 24-Inch...",
    "A lightweight suitcase with good reviews is the Samsonite Lite-Cube...",
    "The suitcase with the highest rating is the TravelPro Platinum Elite...",
    "A durable suitcase for travel is the American Tourister Fieldbrook II...",
    "Suitcases bought frequently last month include the Delsey Helium Aero...",
    "Best seller suitcase options are the Samsonite Winfield 2 20-Inch..."
]

# =====================================================================
# SAFE PRODUCT RETRIEVER
# =====================================================================
def safe_search_product(query, k=5):
    df = search_product(query, k=k)
    if df is None or df.empty:
        df = pd.DataFrame([{
            "title": "",
            "price": 0,
            "rating": 0,
            "reviews": 0,
            "bestseller": False,
            "bought_last_month": 0
        }])
    else:
        df = df.head(k)
    return df

# =====================================================================
# EVALUATION LOOP
# =====================================================================
def evaluate_all_models(rag_pipeline_func):
    model_results = {
        "flan_large": [],
        "microsoft/phi-2": [],
        "flan_small": []
    }

    progress = st.progress(0)

    for q_idx in range(min(len(queries), len(references))):
        user_query = queries[q_idx]
        reference = references[q_idx]

        st.write(f"### Evaluating Query {q_idx+1}: {user_query}")
        outputs = rag_pipeline_func(user_query, top_k=5)

        # retrieve products
        products_list = safe_search_product(user_query, k=5)
        formatted_products = "\n".join(format_products_for_llm(products_list))

        for model_name in ["flan_large", "microsoft/phi-2", "flan_small"]:
            if model_name not in outputs.get("normal", {}) or model_name not in outputs.get("json", {}):
                continue

            pred_text = outputs["normal"][model_name]
            r_text = compute_rouge(pred_text, reference)
            g_text = grounding_rate(pred_text, formatted_products)
            f_text = fluency_score(pred_text)
            final_text = (r_text + g_text + f_text) / 3

            pred_json_str = outputs["json"][model_name]
            try:
                pred_json = json.loads(pred_json_str)
                pred_json_combined = " ".join(str(v) for v in pred_json.values())
            except:
                pred_json_combined = pred_json_str

            r_json = compute_rouge(pred_json_combined, reference)
            g_json = grounding_rate(pred_json_combined, formatted_products)
            f_json = fluency_score(pred_json_combined)
            final_json = (r_json + g_json + f_json) / 3

            model_results[model_name].append({
                "Model": model_name,
                "Query": user_query,
                "Normal Accuracy": final_text,
                "JSON Accuracy": final_json
            })

        progress.progress((q_idx+1) / len(queries))

    return model_results


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.title("RAG Model Evaluation Dashboard")
st.write("Evaluate FLAN-Large, FLAN-Small, Phi-2 on multiple queries.")

# ==========================================================
# PAGE TITLE
# ==========================================================
st.title("Product Recommendation RAG System")
st.write("Ask for product recommendations using 3 FLAN models (Large, Base, Small).")
st.write("---")

# ==========================================================
# USER INPUT
# ==========================================================
user_query = st.text_input("Enter your product query:")

if st.button("Search"):
    if user_query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        st.info("Searching for recommendations...")
        
        # ==========================================================
        # RUN MODELS (NORMAL OUTPUT ONLY)
        # ==========================================================
        output = rag_pipeline_all_models(user_query, top_k=5)

        # ------------------------------------------------------
        # FLAN-LARGE
        # ------------------------------------------------------
        st.subheader("🟦 FLAN-T5-LARGE — Normal Output")
        st.write(output["normal"]["flan_large"])
        st.markdown("---")

        # ------------------------------------------------------
        # FLAN-BASE
        # ------------------------------------------------------
        st.subheader("🟦 FLAN-T5-BASE — Normal Output")
        st.write(output["normal"]["flan_base"])
        st.markdown("---")

        # ------------------------------------------------------
        # FLAN-SMALL
        # ------------------------------------------------------
        st.subheader("🟦 FLAN-T5-SMALL — Normal Output")
        st.write(output["normal"]["flan_small"])
        st.markdown("---")
if st.button("Run Evaluation"):
    results = evaluate_all_models(rag_pipeline_all_models)

    # convert results to dataframe
    rows = []
    for model_name, entries in results.items():
        for e in entries:
            rows.append(e)

    df_results = pd.DataFrame(rows)

    st.subheader("📊 Results Table")
    st.dataframe(df_results)

    # =================================================================
    # CLUSTERED BAR CHART
    # =================================================================
    st.subheader("📈 Accuracy Comparison Chart")

    models = df_results["Model"].unique()
    queries_list = df_results["Query"].unique()

    fig, axes = plt.subplots(1, len(queries_list), figsize=(22, 5), sharey=True)
    colors = {"Normal": "blue", "JSON": "orange"}
    width = 0.35

    for i, query in enumerate(queries_list):
        ax = axes[i]
        data = df_results[df_results["Query"] == query]

        x = np.arange(len(data))
        ax.bar(x - width/2, data["Normal Accuracy"], width, color="blue")
        ax.bar(x + width/2, data["JSON Accuracy"], width, color="orange")

        ax.set_xticks(x)
        ax.set_xticklabels(data["Model"], rotation=45, ha='right')
        ax.set_title(query)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.text(0.04, 0.5, 'Accuracy', rotation='vertical')

    st.pyplot(fig)
