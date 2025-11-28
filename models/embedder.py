import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load Data
df = pd.read_excel("data/data.xlsx")
cat_df = pd.read_csv("data/amazon_categories.csv")

df = df.merge(cat_df, left_on="category_id", right_on="id", how="left")
df = df.dropna()

df["price"] = df["price"].replace("$","",regex=True).astype(float)
df = df[df['price'] > 0]

df = df.merge(cat_df, left_on="category_id", right_on="id", how="left")

df['short_text'] = (
    df['title'].astype(str) + " | Category: " + df['category_name_x'].astype(str)
)

df['text'] = (
    "Title: " + df['title'].astype(str) +
    " | Category: " + df['category_name_y'].astype(str) +
    " | Price: $" + df['price'].astype(str) +
    " | List Price: $" + df['listPrice'].astype(str) +
    " | Rating: " + df['stars'].astype(str) +
    " | Reviews: " + df['reviews'].astype(str) +
    " | Best Seller: " + df['isBestSeller'].astype(str) +
    " | Bought Last Month: " + df['boughtInLastMonth'].astype(str) +
    " | Product URL: " + df['productURL'].astype(str) +
    " | Image URL: " + df['imgUrl'].astype(str)
)

# Embedding Model
embed_model = SentenceTransformer('all-MiniLM-l6-v2')

embeddings = embed_model.encode(
    df['text'].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
).astype('float32')

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

def search_product(query, k=5):
    query_vec = embed_model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, k)
    return df.iloc[indices[0]]
