#!/usr/bin/env python3
import requests
import sys
import time
import queue
import threading
from tqdm import tqdm
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed # !!! only use if using bulk insertion and processing of vectors

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
max_workers=25
RATE_LIMIT = 0   # 500 ms
embed_q = queue.Queue()
store_q = queue.Queue()
progress_bar = None


# ----------------------------
# 3rd Party embedding api from huggingface space
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


# inserting data into input queue for embedding creation
def bulk_process(avengers_texts_large):
    for line in avengers_texts_large:
        embed_q.put(line)   # fast producer

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



avengers_texts_large = [
            "Iron Man builds a new suit to combat threats.",
            "Captain America wields his vibranium shield.",
            "Thor summons lightning with Mjolnir.",
            "Black Widow infiltrates enemy bases.",
            "Hawkeye aims his arrows with deadly accuracy.",
            "The Avengers assemble to stop global catastrophes.",
            "Loki plots mischief in Asgard and Midgard.",
            "Thanos seeks the Infinity Stones to reshape the universe.",
            "Doctor Strange manipulates time to prevent disasters.",
            "Spider-Man swings through New York fighting crime.",
            "Black Panther protects Wakanda and its secrets.",
            "Scarlet Witch struggles with her chaotic powers.",
            "Vision integrates the Mind Stone within himself.",
            "The Hulk grows stronger with each transformation.",
            "Ant-Man shrinks and grows using Pym Particles.",
            "Wasp fights alongside Ant-Man with agility and stingers.",
            "Nick Fury recruits heroes to form the Avengers initiative.",
            "The Battle of New York is the Avengers' first big fight.",
            "The Snap by Thanos wipes out half the universe.",
            "Captain Marvel arrives to aid in cosmic battles.",
            "The Time Heist retrieves Infinity Stones from the past.",
            "Hawkeye becomes Ronin to fight crime alone.",
            "Thor loses his hammer but discovers inner strength.",
            "Loki switches between villain and anti-hero roles.",
            "Ultron threatens humanity and must be stopped.",
            "Vision and Scarlet Witch share a complex bond.",
            "Wakanda provides advanced technology to the Avengers.",
            "Spider-Man joins Doctor Strange in mystical battles.",
            "Black Panther showcases unparalleled combat skills.",
            "Iron Man sacrifices himself to save the universe.",
            "Captain America returns the Infinity Stones to their timelines.",
            "The multiverse introduces alternate realities and threats.",
            "Hawkeye trains his daughter in archery and combat.",
            "Spider-Man faces the responsibilities of being a hero.",
            "Thor controls lightning without Mjolnir during battles.",
            "Scarlet Witch learns to harness her full powers.",
            "The Guardians of the Galaxy assist in universal threats.",
            "Ant-Man explores the Quantum Realm to alter events.",
            "Nick Fury coordinates secret missions to protect Earth.",
            "Doctor Strange opens portals to fight interdimensional threats.",
            "The Hulk learns to control his anger and transformations.",
            "Black Widow undertakes covert S.H.I.E.L.D. missions.",
            "Thanos' army challenges Earth's mightiest heroes.",
            "The Avengers must cooperate to defeat powerful enemies.",
            "Iron Man builds new suits to tackle evolving threats.",
            "Captain America leads with courage and strategy.",
            "Thor faces cosmic challenges beyond Asgard.",
            "Black Panther defends Wakanda while aiding global fights.",
            "Spider-Man encounters alien invasions in the city.",
            "Scarlet Witch confronts her dark past to control chaos.",
            "Vision sacrifices himself for the greater good.",
            "Ant-Man navigates the Microverse to save the day.",
            "Hawkeye mentors future generations of heroes.",
            "The Avengers celebrate victories and mourn losses.",
            "Doctor Strange trains new sorcerers to defend reality.",
            "Iron Man mentors Spider-Man in heroics and tech.",
            "Captain America retires after decades of service.",
            "Thor journeys across the cosmos to restore balance.",
            "Black Widow uncovers hidden conspiracies threatening Earth.",
            "Loki’s schemes create unexpected alliances and conflicts.",
            "The Avengers confront cosmic entities beyond Earth.",
            "Spider-Man faces moral dilemmas as a teenage hero.",
            "Black Panther forges alliances to protect Wakanda and the world.",
            "Scarlet Witch’s powers grow stronger with training and experience.",
            "Hulk works with Bruce Banner to find inner peace.",
            "Ant-Man teams up with the Wasp to combat threats.",
            "The Infinity Gauntlet determines the fate of all beings.",
            "The Avengers prevent villains from exploiting alternate universes.",
            "Nick Fury monitors threats from the shadows.",
            "Doctor Strange uses magic to maintain the balance of reality.",
            "Thor battles cosmic forces threatening multiple worlds.",
            "Iron Man's legacy inspires the next generation.",
            "Captain America mentors young soldiers in courage and ethics.",
            "Black Panther protects vibranium while aiding global conflicts.",
            "Hawkeye uses skill and strategy in every mission.",
            "Scarlet Witch faces adversaries that challenge her morality.",
            "Spider-Man balances everyday life with superhero duties.",
            "The Avengers unite across galaxies to face universal threats.",
            "Ant-Man invents new technologies to enhance combat effectiveness.",
            "Vision and Scarlet Witch fight to preserve universal balance.",
            "Thor adapts to challenges in a universe without Mjolnir.",
            "Loki manipulates multiverse chaos for his schemes.",
            "The Avengers’ teamwork is tested in extreme battles.",
            "Iron Man designs specialized suits for different threats.",
            "Captain America rallies the team in moments of crisis.",
            "Thor faces battles that span galaxies.",
            "Black Widow handles espionage and covert operations.",
            "Hawkeye tracks enemies with pinpoint accuracy.",
            "Scarlet Witch navigates her chaotic powers responsibly.",
            "Spider-Man encounters villains from different dimensions.",
            "The Guardians of the Galaxy team up with the Avengers.",
            "Doctor Strange combats threats from other realities.",
            "Ant-Man explores shrinking technology for tactical advantages.",
            "Vision integrates logic and emotion in critical decisions.",
            "Thor learns humility while leading in Asgardian conflicts.",
            "Loki forms temporary alliances to achieve his goals.",
            "The Avengers protect Earth from cosmic-scale dangers.",
            "Iron Man improvises solutions during desperate situations.",
            "Captain America inspires hope and resilience in teammates.",
            "Black Panther leverages Wakandan resources for global defense.",
            "Hawkeye’s precision tips the scales in crucial missions.",
            "Scarlet Witch adapts her powers for maximum effectiveness.",
            "Spider-Man’s agility and intellect help him overcome threats.",
            "The Avengers coordinate multi-front strategies.",
            "Ant-Man’s ingenuity contributes to complex operations.",
            "Vision evaluates scenarios with logic and ethics.",
            "Thor confronts enemies with both strength and wisdom.",
            "Loki’s cunning complicates the Avengers’ plans.",
            "Doctor Strange manipulates time and space strategically.",
            "Black Widow neutralizes high-level threats covertly.",
            "Hawkeye trains new recruits in strategy and combat.",
            "Iron Man collaborates with allies to tackle evolving dangers.",
            "Captain America balances leadership with frontline action.",
            "Spider-Man learns from older heroes to improve his tactics.",
            "Scarlet Witch reconciles past mistakes to strengthen resolve.",
            "Thor builds alliances across realms for critical missions.",
            "The Avengers respond rapidly to multiversal crises.",
            "Ant-Man uses shrinking technology to execute daring plans.",
            "Vision’s ethical judgment guides key team decisions.",
            "Loki’s mischief occasionally aids the Avengers unknowingly.",
            "Black Panther demonstrates mastery in combat and diplomacy.",
            "Doctor Strange advises the team on mystical threats.",
            "Hawkeye executes precise strikes in high-stakes scenarios.",
            "Iron Man develops countermeasures against advanced threats.",
            "Captain America strategizes to maintain team cohesion.",
            "Spider-Man improvises to overcome unpredictable obstacles.",
            "Scarlet Witch channels her power responsibly during battles.",
            "Thor’s strength and experience turn the tide in fights.",
            "The Avengers coordinate to prevent universal collapse.",
            "Ant-Man discovers new solutions within the Quantum Realm.",
            "Vision’s presence ensures ethical decision-making.",
            "Loki occasionally switches sides when it serves a greater good.",
            "Black Widow investigates hidden threats across the globe.",
            "Hawkeye monitors and eliminates high-value targets.",
            "Iron Man mentors younger heroes in both technology and courage.",
            "Captain America inspires unity among diverse team members.",
            "Spider-Man balances personal life with heroic responsibilities.",
            "Scarlet Witch faces internal and external challenges with resolve.",
            "Thor confronts threats from both cosmic and earthly origins.",
            "The Avengers maintain vigilance across all dimensions.",
            "Ant-Man uses inventive tactics to complement the team.",
            "Vision provides guidance and support in complex missions.",
            "Loki’s cunning requires constant vigilance from the Avengers.",
            "Black Panther ensures Wakanda remains a bastion of strength.",
            "Doctor Strange protects reality from mystical and interdimensional threats.",
            "Hawkeye demonstrates expertise in reconnaissance and precision strikes.",
            "Iron Man’s innovations safeguard the team against emerging dangers.",
            "Captain America exemplifies leadership in times of crisis.",
            "Spider-Man contributes with agility, intellect, and ingenuity.",
            "Scarlet Witch hones her abilities to protect friends and the universe.",
            "Thor commands the power of thunder and lightning to combat enemies.",
            "The Avengers unify to face threats too great for any hero alone.",
            "Ant-Man explores new applications of shrinking and growth technology.",
            "Vision evaluates and resolves conflicts with calm reasoning.",
            "Loki’s schemes intersect unpredictably with the Avengers’ missions.",
            "Black Widow operates covertly to neutralize secret threats.",
            "Hawkeye ensures mission objectives are achieved with precision.",
            "Iron Man adapts to evolving threats with creativity and technology.",
            "Captain America balances strategic oversight with direct action.",
            "Spider-Man learns and grows from interactions with experienced heroes.",
            "Scarlet Witch channels her powers with focus and responsibility.",
            "Thor leads battles with courage, experience, and strength.",
            "The Avengers’ teamwork allows them to succeed where individuals fail.",
            "Ant-Man discovers creative solutions within challenging situations.",
            "Vision assists in planning and executing critical operations.",
            "Loki’s unpredictable actions create both challenges and opportunities.",
            "Black Panther leverages Wakanda’s technology and resources effectively.",
            "Doctor Strange advises the team on mystical and cosmic phenomena."
        ]

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

    total_items = len(avengers_texts_large)
    progress_bar = tqdm(total=total_items, desc="Embedding", ncols=100)

    start_time = time.time()

    start_workers()
    bulk_process(avengers_texts_large)

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
