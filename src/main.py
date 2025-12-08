#!/usr/bin/env python3
import requests
import sys
import time
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# ----------------------------
# external inference endpoints
# ----------------------------
EMBED_URL = "https://lamhieu-lightweight-embeddings.hf.space/v1/embeddings" # your bge-m3 embed API
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


# ----------------------------
# api helpers
# ----------------------------
def embed(text: str):
    # custom headers applicable for above inference api
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': 'bge-m3',
        'input': text,
    }

    r = requests.post(EMBED_URL, headers=headers, json=json_data)
    r.raise_for_status()
    # print(r.json())
    return r.json()["data"][0]["embedding"]

# def llm_suggest(text: str, k: int = 3):
#     try:
#         r = requests.post(SUGGEST_URL, json={"text": text, "k": k})
#         r.raise_for_status()
#         return r.json().get("sentences", [])
#     except Exception:
#         return []

# ----------------------------
# mongo helpers
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
# storing to atlas (memory-building)
# ----------------------------
def store_sentence(text: str, emb):
    coll.insert_one({
        "text": text,
        "embedding": emb,
        "ts": time.time()
    })

# ----------------------------
# main loop
# ----------------------------
def main():
    print("Realtime sentence recommender (bge-m3, Atlas vector search)")
    print("type word → press Enter\n")

    while True:
        seed = input("seed> ").strip()
        if not seed:
            continue

        # 1) embed
        emb = embed(seed)

        # 2) vector lookup in atlas
        db_hits = vector_query(emb)

        # 3) optional generative suggestions
        # llm_hits = llm_suggest(seed, k=3)

        # 4) merge + print
        print("\n--- matches from DB ---")
        for d in db_hits:
            print(f"[DB] {d['text']}")

        # print("\n--- generated suggestions ---")
        # for s in llm_hits:
        #     print(f"[GEN] {s}")

        print()

        # 5) optional: store new sentence (self-growing memory)
        store_sentence(seed, emb)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

# or realtime char by char (no return detection) then add below one

# from readchar import readchar
# import threading

# buffer = ""
# last = time.time()

# def loop():
#     global buffer, last
#     while True:
#         c = readchar()
#         if c == '\n':
#             print()  # newline
#             handle(buffer)
#             buffer = ""
#         else:
#             buffer += c
#             now = time.time()
#             if now - last > 0.25:   # 250ms debounce
#                 handle(buffer)
#             last = now



#  for doing this more sort of in asynchornous way then check below
# import asyncio
# import aiohttp
# import aioconsole
# from motor.motor_asyncio import AsyncIOMotorClient

# ----------------------------
# external inference endpoints
# ----------------------------
# CONN_URL = "http://localhost:8000/embed"
# SUGGEST_URL = "http://localhost:8000/suggest"

# ----------------------------
# mongo config
# ----------------------------
# CONN_URL = "mongodb+srv://..."
# DB_NAME = "recs"
# COLL_NAME = "sentences"
# VECTOR_INDEX = "default"
# VECTOR_LIMIT = 5

# ----------------------------
# HTTP session (global)
# ----------------------------
# session = None

# ----------------------------
# API wrappers (async)
# ----------------------------
# async def embed(text: str):
#     async with session.post(EMBED_URL, json={"text": text}) as r:
#         r.raise_for_status()
#         data = await r.json()
#         return data["embedding"]

# async def suggest(text: str, k: int = 3):
#     try:
#         async with session.post(SUGGEST_URL, json={"text": text, "k": k}) as r:
#             r.raise_for_status()
#             data = await r.json()
#             return data.get("sentences", [])
#     except:
#         return []

# ----------------------------
# Atlas wrappers (async)
# ----------------------------
# async def vector_query(coll, vec):
#     pipeline = [
#         {
#             "$vectorSearch": {
#                 "index": VECTOR_INDEX,
#                 "path": "embedding",
#                 "queryVector": vec,
#                 "numCandidates": VECTOR_LIMIT * 20,
#                 "limit": VECTOR_LIMIT
#             }
#         },
#         {
#             "$project": {
#                 "_id": 0,
#                 "text": 1,
#                 "score": {"$meta": "vectorSearchScore"}
#             }
#         }
#     ]
#     return [d async for d in coll.aggregate(pipeline)]

# optional DB memory growth
# async def store_sentence(coll, text, emb):
#     await coll.insert_one({
#         "text": text,
#         "embedding": emb,
#         "ts": asyncio.get_event_loop().time()
#     })

# ----------------------------
# unified handler
# ----------------------------
# async def handle(seed, coll):
#     # run embedding + suggestions concurrently
#     emb_task = asyncio.create_task(embed(seed))
#     sug_task = asyncio.create_task(suggest(seed, 3))
#     emb = await emb_task
#     llm_hits = await sug_task

#     # run vector search
#     db_hits = await vector_query(coll, emb)

#     # print results
#     print("\n--- DB matches ---")
#     for d in db_hits:
#         print(f"[DB] {d['text']}")

#     print("\n--- Generated suggestions ---")
#     for s in llm_hits:
#         print(f"[GEN] {s}")

#     print()

    # optional: store memory
    # await store_sentence(coll, seed, emb)

# ----------------------------
# main async loop
# ----------------------------
# async def main():
#     global session

#     session = aiohttp.ClientSession()

#     client = AsyncIOMotorClient(CONN_URL)
#     coll = client[DB_NAME][COLL_NAME]

#     print("Async realtime sentence recommender (bge-m3 + Atlas)")
#     print("Type a word → Enter\n")

#     # non-blocking console: aioconsole
#     while True:
#         seed = await aioconsole.ainput("seed> ")
#         seed = seed.strip()
#         if not seed:
#             continue
#         try:
#             await handle(seed, coll)
#         except KeyboardInterrupt:
#             break

#     await session.close()

# if __name__ == "__main__":
#     asyncio.run(main())

