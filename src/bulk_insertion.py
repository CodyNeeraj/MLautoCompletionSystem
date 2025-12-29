import requests
import sys
import time
import queue
import threading
from tqdm import tqdm
import os
import embedding_generator
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed # !!! only use if using bulk insertion and processing of vectors

# ----------------------------
# external inference endpoints
# ----------------------------
EMBED_URL = "<ENTER YOUR API HERE>" # if connecting using any api
# SUGGEST_URL = "http://localhost:8000/suggest" # optional

# ----------------------------
# mongo config
# ----------------------------
load_dotenv()  # loads variables from .env into os.environ

URI_PROTOCOL = "mongodb+srv://"
DB_NAME = "MLautoCompletionSystem"
COLL_NAME = "embeddings-collection"
VECTOR_INDEX = "default"     # must match your Atlas vector index name
VECTOR_LIMIT = 5
mongo_connection_url = os.getenv("mongo_connection_url")
user_pass= os.getenv("user_pass")
user_name=os.getenv("user_name")
CONN_URL=f"{URI_PROTOCOL}{user_name}:{user_pass}{mongo_connection_url}"
max_workers=25
RATE_LIMIT = 0   # 500 ms
embed_q = queue.Queue()
store_q = queue.Queue()
progress_bar = None
_bulk_load_texts = None

# ----------------------------
# 3rd Party embedding api from huggingface space
# ----------------------------
def embed(text: str):
    # # custom headers applicable for above inference api
    # headers = {
    #     'accept': 'application/json',
    #     'Content-Type': 'application/json',
    # }

    # json_data = {
    #     'model': 'bge-m3',
    #     'input': text,
    # }

    # r = requests.post(EMBED_URL, headers=headers, json=json_data)
    # r.raise_for_status()
    # # print(r.json())
    # return r.json()["data"][0]["embedding"]
    return embedding_generator.get_embedding(text)


# Embed Worker implementation to run under a thread instance ran below and do the embedding operation
def embed_worker():
    while True:
        line = embed_q.get()
        if line is None:
            break
        try:
            vec = embed(line)
            store_q.put((line, vec))

            # progress tick (only from embed stage)
            if progress_bar:
                progress_bar.update(1)

        except Exception as e:
            print("embed error:", e)
        finally:
            time.sleep(RATE_LIMIT)  # respect rate limit
            embed_q.task_done() # marking the current item in queue done 

# Embed database Worker implementation to run under a thread instance ran below and do the database insert operation (parrallely) under the multiple threads
def db_worker():
    while True:
        item = store_q.get()
        if item is None:
            break
        try:
            text, vec = item
            store_sentence(text, vec)
            # optional logging:
            # print(f"[DB] stored: {text[:40]}")
        except Exception as e:
            print("db error:", e)
        finally:
            store_q.task_done()

# def db_worker_inner(item):
#     text, vec = item
#     try:
#         store_sentence(text, vec)
#         print(f"[DB] stored: {text[:50]}")
#     except Exception as e:
#         print("db error:", e)


# create threads based on the producer/consumers queues
def start_workers():
    # one embed worker
    threading.Thread(target=embed_worker, daemon=True).start()

    # many db workers
    for _ in range(max_workers):
        threading.Thread(target=db_worker, daemon=True).start()


# inserting file loaded data into input queue for embedding creation
def ingest_texts():
    global _bulk_load_texts

    # Gets the script's own directory and appends the relative path
    script_dir = Path(__file__).parent
    file_path = script_dir / "raw_text" / "avengers.txt"

    try:
        # Open and read the local text file into a list
        with open(file_path, "r", encoding="utf-8") as f:
            _bulk_load_texts = [line.strip() for line in f if line.strip()]

        for line in _bulk_load_texts:
            embed_q.put(line)   # fast producer
    
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}. "
              "Ensure you are running the script from the parent folder of 'raw_text'.")

# ----------------------------
# Initilization of Database/Collection
# ----------------------------
client = MongoClient(CONN_URL)
coll = client[DB_NAME][COLL_NAME]

# Vector aggregation search on a MongoDB collection using a vector index
# (or general indexes) via a pipeline. This scans a given embedding array
# using an upstream-created vector index. The 'index' parameter specifies
# the name of the vector index, and the 'path' parameter specifies the
# document field over which the index was created.

def vector_query(vec):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", #name of the created index here
                "path": "embedding", #name of document field
                "queryVector": vec,
                "numCandidates": VECTOR_LIMIT * 20,
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
    # print(list(coll.aggregate(pipeline)))
    return list(coll.aggregate(pipeline))

# ----------------------------
# storing embedding to atlas (for memory-building)
# ----------------------------
def store_sentence(text: str, emb):
    coll.insert_one({
        "text": text,
        "embedding": emb,
        "ts": time.time()
    })

# Non usable Sequential function
# def bulk_process_threading(line):
#     try:
#         processed_vector = embed(line) #creating embeddings
#         store_sentence(line, processed_vector) # inserting the data
#     except Exception as e:
#         print(e)       


# ----------------------------
# main loop
# ----------------------------
def main():
    global progress_bar

    print("Starting the bulk insertion operation")

    # total_items = len(_bulk_load_texts)
    progress_bar = tqdm(total=168, desc="Embedding", ncols=100)

    start_time = time.time()

    start_workers()
    ingest_texts()

    # wait
    embed_q.join()
    store_q.join()

    end_time = time.time()
    progress_bar.close()

    print(f"\nCompleted in {end_time - start_time:.2f} seconds.")
  

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
