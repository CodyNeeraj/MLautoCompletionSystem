import json
import os
import time, requests
from pathlib import Path
from dotenv import load_dotenv
from newsdataapi import NewsDataApiClient

load_dotenv()
# Configuration
API_KEY = os.getenv("news_api_key")
FILE_PATH = Path(__file__).parent / "raw_text" / "news_queue.json"
KEYWORDS = "bitcoin OR crypto OR markets OR technology" 
COUNTRY="us,in,de,gb,ru"
LANGUAGE="en"
CATEGORY="business,technology,world,top,health"
TZ="asia/kolkata"

api = NewsDataApiClient(apikey=API_KEY)

def update_local_storage(new_articles):
    # Load existing data to prevent duplicates
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    initial_count = len(data)
    
    # processing json object into small parsable objects
    for art in new_articles:
        # Use article_id as unique ID to avoid duplicates
        art_id = art.get("article_id")
        if art_id not in data:
            data[art_id] = {
                "title": art.get("title"),
                "description": art.get("description"),
                "url": art.get("link"),
                "pubDate": art.get("pubDate"),
                "processed": False  # Flag for your inference script
            }

    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Added {len(data) - initial_count} new articles. Total present in file queue: {len(data)}")

def run_ingester():
    query_params = {
        "apikey": API_KEY,
        "country": COUNTRY,
        "language": LANGUAGE,
        "category": CATEGORY,
        "timezone": TZ,
        "prioritydomain": "top",
        "image": 0,
        "video": 0,
        "removeduplicate": 1
    }
    try:
        # Fetching latest news
        response = requests.get("https://newsdata.io/api/1/latest", params=query_params)
        if response.status_code == 200:
            data = response.json()
            update_local_storage(data.get("results", []))
    except Exception as e:
        print(f"Error fetching news: {e}")

if __name__ == "__main__":
    # You can set this to run every 5 seconds
    while True:
        run_ingester()
        print("Waiting 5 seconds...")
        time.sleep(5)  # Pause execution for 5 seconds

