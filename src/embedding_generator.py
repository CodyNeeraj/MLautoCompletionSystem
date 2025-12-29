import streamlit as st
# import chromadb
from sentence_transformers import SentenceTransformer
from st_keyup import st_keyup

# came from hf download BAAI/bge-small-en-v1.5 --local-dir ./models/bge-small
DISK_PATH =  "./models/bge-small"

# 'cuda' ensures it uses VRAM for high-speed inference
# @st.cache_resource
def load_model():
    return SentenceTransformer(DISK_PATH, device='cuda', trust_remote_code=True)

# Set the global flagfor model object initialization
_model = None

def get_model():
    global _model
    # print("--- Loading model into VRAM ---")
    if _model is not None:
        # print("--- ✨ Model retrieved from RAM Cache (already Warm) ---")
        return _model
    _model = load_model()
    return _model

# 2. Setup Persistent ChromaDB
# This creates a folder 'my_vector_db' with a .sqlite file inside
# client = chromadb.PersistentClient(path="./my_vector_db")
# collection = client.get_or_create_collection(name="recommendations")

def get_embedding(text: str, model_instance=None):
    """
    Utility function to create embeddings.
    Can be imported and used in other files.
    """
    if not text or not text.strip():
        return []
    
    # If no model provided, use the cached global one
    m = get_model()
    
    # Generate embedding (normalize_embeddings is recommended for BGE)
    return m.encode(text, normalize_embeddings=True).tolist()

def run_ui():
    st.title("Fast Semantic Search")
    query = st_keyup("Search for something...", key="interactive_input")

    if query:
        # High-speed embedding generation
        query_embedding = get_embedding(query)
        print(query_embedding)
        # st.code(query_embedding, language='python')

        with st.status("Generating embedding...") as status:
            st.write("### Query Embedding Vector")
            st.write(query_embedding[:10])  # Displays as a scrollable list/array
            status.update(label="Embedding complete!", state="complete")

        # with st.expander("View Raw Embedding"):
        #     st.write(f"Vector Dimensions: {len(query_embedding)}")
        #     st.json(query_embedding)  # st.json provides a better interactive view for lists
        
    #     # Query ChromaDB
        # results = collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=5
    #     )
    #     st.write(results['documents'])
    
if __name__ == "__main__":
    model = load_model()
    run_ui()