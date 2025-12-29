import streamlit as st
import embedding_generator
from pymongo import MongoClient
import os
from dotenv import load_dotenv, find_dotenv
from st_keyup import st_keyup

load_dotenv() 

URI_PROTOCOL = "mongodb+srv://"
DB_NAME = "MLautoCompletionSystem"
COLL_NAME = "embeddings-collection"
VECTOR_INDEX = "vector_index_embeddings_key"     # must match your Atlas vector index name
VECTOR_LIMIT = 10
mongo_connection_url = os.getenv("mongo_connection_url")
user_pass= os.getenv("user_pass")
user_name=os.getenv("user_name")
CONN_URL=f"{URI_PROTOCOL}{user_name}:{user_pass}{mongo_connection_url}"

@st.cache_resource
def init_connection():
    load_dotenv(find_dotenv()) 
    print(CONN_URL)
    client = MongoClient(CONN_URL)
    return client

client = init_connection()
db = client[DB_NAME]
collection = db[COLL_NAME]

st.set_page_config(
    page_title="Lightning Semantic Search", 
    page_icon="🤖",
    layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.9rem; /* Adjust this value as needed (default is ~6rem) */
    }
    </style>
    """, unsafe_allow_html=True)

st.header("🚀 Sentence Recommendation System")

def run_ui():
    # st.title("Lightning Semantic Search")
    query = st_keyup("Search for something...", key="interactive_input")

    if query:
        query_embedding = embedding_generator.get_embedding(query)
        results = vector_query(query_embedding)

        with st.expander("Realtime Generated Embeddings Statistics"):
            st.write(f"Vector Dimensions: {len(query_embedding)}")
            st.write(f"Length of Find Result: {len(results)}")
            st.json(results) 

        with st.spinner("Searching MongoDB Atlas..."):
            if results:
                # We iterate through the results which are already sorted descending by MongoDB
                # for item in results:
                #     score = item["score"]
                #     text = item["text"]
                #     st.markdown(f"**Score:** `{score:.4f}` - {text}")

                st.dataframe(
                results,
                column_order=("score", "text"),
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "Relevance",
                        help="Vector Search Similarity Score",
                        format="%.4f",
                        min_value=0.0,
                        max_value=1.0,
                        color="auto"
                    ),
                    "text": st.column_config.TextColumn("Matched Results", width="large"),
                    "embedding": None
                },
                hide_index=True,
                use_container_width=True
                )
            else:
                st.info("No results found for your query.")

def vector_query(vec):
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX, #name of the created index here
                "path": "embedding", #name of document field
                "queryVector": vec,
                "numCandidates": VECTOR_LIMIT * 30,
                "limit": VECTOR_LIMIT
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "embedding": 1,  # include the document field to be scanned
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    return list(collection.aggregate(pipeline))

if __name__ == "__main__":
    run_ui()