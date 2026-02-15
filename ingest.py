import os
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Load the secret connection string from your .env file
load_dotenv()
mongo_uri = os.getenv("MONGODB_URI")

if not mongo_uri:
    print("Error: MONGODB_URI not found. Check your .env file!")
    exit()

print("Connecting to MongoDB...")

# 2. Connect to the Database
try:
    client = pymongo.MongoClient(mongo_uri)
    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

# 3. Define the Database and Collection names
db = client["voice_agent_db"]       # The database name
collection = db["knowledge_base"]   # The collection (folder) for data

# 4. Initialize the Free AI Model (converts text to numbers)
print("Loading the AI model (this allows the bot to understand meaning)...")
# This downloads a small, free model from HuggingFace
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

# 5. The Data to Upload (You can change this later!)
texts = [
    "The user is building a Real-Time Voice AI Agent using Python.",
    "This project uses MongoDB Vector Search to find relevant information quickly.",
    "We are using free tools like Groq for LLM and Edge TTS for voice.",
    "The project focuses on low latency (speed) for a natural conversation."
]

print("Processing and uploading data...")

# 6. Convert Text to Vectors and Upload
for text in texts:
    # Convert the sentence into a list of numbers (vector)
    vector = embedder.encode(text).tolist()
    
    # Create the data packet
    document = {
        "text": text,
        "embedding": vector
    }
    
    # Insert into MongoDB
    collection.insert_one(document)

print("✅ Success! Data uploaded to MongoDB.")