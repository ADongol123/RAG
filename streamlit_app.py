import streamlit as st
from models.rag_pipeline import rag_pipeline_all_models

st.title("Product Recommendation RAG System")
st.write("Ask for product recommendations based on your query.")

user_query = st.text_input("Enter your product query:")

if st.button("Search"):
    if user_query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        st.info("Searching for recommendations...")
        output = rag_pipeline_all_models(user_query, top_k=5)
        
        st.subheader("FLAN-T5-LARGE (NORMAL)")
        st.write(output["normal"]["flan_large"])
        
        
        st.subheader("FLAN-T5-BASE (NORMAL)")
        st.write(output["normal"]["flan_base"])
        
        st.subheader("FLAN-T5-SMALL (NORMAL)")
        st.write(output["normal"]["flan_small"])
        