from pymongo import MongoClient
from itertools import islice
import os
from dotenv import load_dotenv

load_dotenv() 

URI_PROTOCOL = "mongodb+srv://"
mongo_connection_url = os.getenv("mongo_connection_url")
user_pass= os.getenv("user_pass")
user_name=os.getenv("user_name")
MONGO_URI   = f"{URI_PROTOCOL}{user_name}:{user_pass}{mongo_connection_url}"
DB_NAME     = "MLautoCompletionSystem"
COLL_NAME   = "embeddings-collection"
DEDUP_FIELD = "text"     # field that must be unique
BATCH_SIZE  = 5000               # safe delete batch size

client = MongoClient(MONGO_URI)
coll   = client[DB_NAME][COLL_NAME]

def batch(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk

print("Scanning for duplicates...")

cursor = coll.aggregate([
    {"$group": {
        "_id": f"${DEDUP_FIELD}",
        "ids": {"$push": "$_id"},
        "count": {"$sum": 1}
    }},
    {"$match": {"count": {"$gt": 1}}}
])

total_deleted = 0

for group in cursor:
    # keep first, delete the rest
    dup_ids = group["ids"][1:]
    for chunk in batch(dup_ids, BATCH_SIZE):
        result = coll.delete_many({"_id": {"$in": chunk}})
        total_deleted += result.deleted_count

print(f"Done. Deleted {total_deleted} duplicate documents.")