#!/usr/bin/env python3
import requests
import sys
import time
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


def bulk_process_threading(line):
    try:
        processed_vector = embed(line) #creating embeddings
        store_sentence(line, processed_vector) # inserting the data
    except Exception as e:
        print(e)       


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
        store_sentence(seed,emb)


        # Threading implementation with Futures
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(bulk_process_threading, line) for line in avengers_texts_large]
            # for future in as_completed(futures):
            #     print(f"Processed: {future.result()}")

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

