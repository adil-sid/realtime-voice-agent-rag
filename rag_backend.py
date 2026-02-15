import os
import pymongo
from dotenv import load_dotenv
import fitz  # This is PyMuPDF (The stronger PDF reader)
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# --- Database & AI Setup ---
def get_mongo_collection():
    """Connects to MongoDB"""
    try:
        client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        return client["voice_agent_db"]["knowledge_base"]
    except Exception as e:
        print(f"❌ MongoDB Error: {e}")
        return None

def get_embedder():
    """Loads the AI model"""
    return SentenceTransformer('all-mpnet-base-v2')

# --- Robust PDF Processing ---
def extract_text_from_pdf(pdf_file):
    """
    Reads text using PyMuPDF (fitz).
    Handles complex layouts and weird fonts much better.
    """
    try:
        # Streamlit passes a "UploadedFile" object (bytes)
        # We need to read those bytes into PyMuPDF
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        
        if len(text) < 50:
            print("⚠️ Warning: Extracted text is very short. Is this a scanned image?")
            
        return text
    except Exception as e:
        print(f"❌ PDF Read Error: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping pieces for better context"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Cleanup whitespace (tabs, newlines)
        clean_chunk = " ".join(chunk.split())
        
        if len(clean_chunk) > 20: # Only keep meaningful chunks
            chunks.append(clean_chunk)
            
        start += chunk_size - overlap # Overlap logic
        
    return chunks

# --- Main Ingestion Function ---
def process_and_store_pdf(uploaded_file):
    """
    1. Reads PDF
    2. Chunks it
    3. Embeds it (Vectors)
    4. Saves to MongoDB
    """
    collection = get_mongo_collection()
    if collection is None:
        return 0

    embedder = get_embedder()
    
    # 1. Read
    print(f"📖 Reading {uploaded_file.name}...")
    raw_text = extract_text_from_pdf(uploaded_file)
    
    if not raw_text:
        print("❌ No text extracted. The PDF might be empty or scanned.")
        return 0
    
    # 2. Chunk
    chunks = chunk_text(raw_text)
    print(f"🧩 Split into {len(chunks)} chunks.")
    
    # 3. Embed & Store
    documents = []
    
    for i, chunk_text_content in enumerate(chunks):
        # Convert text to numbers (Vector)
        vector = embedder.encode(chunk_text_content).tolist()
        
        doc = {
            "file_name": uploaded_file.name,
            "chunk_id": i,
            "text": chunk_text_content,
            "embedding": vector
        }
        documents.append(doc)
    
    if documents:
        # Optional: Delete old entries for this specific file to avoid duplicates
        collection.delete_many({"file_name": uploaded_file.name})
        
        collection.insert_many(documents)
        print(f"✅ Successfully stored {len(documents)} chunks.")
        return len(documents)
        
    return 0

def clear_database():
    """Wipes the memory clean"""
    coll = get_mongo_collection()
    if coll is not None:
        coll.delete_many({})
        return True
    return False