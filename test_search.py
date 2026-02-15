import os
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. Connect
client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
db = client["voice_agent_db"]
collection = db["knowledge_base"]

# 2. Load the same AI model
print("Loading AI model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Simulate a User Question
query = "What tools are used in this project?"
print(f"\nUser Question: '{query}'")

# 4. Convert question to vector
query_vector = embedder.encode(query).tolist()

# 5. Search MongoDB (The "Retrieval" step)
print("Searching database...")
results = collection.aggregate([
    {
        "$vectorSearch": {
            "index": "vector_index",      # IMPORTANT: Check if your index name is 'default' or 'vector_index'
            "path": "embedding",     # The field we search
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": 2
        }
    }
])

# 6. Show Results
found = False
for doc in results:
    found = True
    print(f"\nFound Document:\n -> {doc['text']}")

if not found:
    print("\n❌ No results found. Did you create the Search Index on MongoDB Atlas?")