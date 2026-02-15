import os
import pymongo
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

def get_mongo_collection():
    """Connets to MongoDB"""
    try:
        client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        return client["voice_agent_db"]["knowledge_base"]
    except Exception as e:
        print(f"MongoDB Error: {e}")
        return None
    
def get_embedder():
    """Loads AI model"""
    return SentenceTransformer('llama-3.1-8b-instant')

def extract_text_from_pdf(pdf_file):
    """Reads text from a Streamlit uploaded file"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping pieces for better context"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Cleanup whitespace
        clean_chunk = chunk.replace('\n', ' ').strip()
        
        if len(clean_chunk) > 10: # Ignore empty chunks
            chunks.append(clean_chunk)
            
        start += chunk_size - overlap # Overlap logic
        
    return chunks

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
    raw_text = extract_text_from_pdf(uploaded_file)
    
    # 2. Chunk
    chunks = chunk_text(raw_text)
    
    # 3. Embed & Store
    documents = []
    print(f"🚀 Processing {len(chunks)} chunks...")
    
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
        collection.insert_many(documents)
        return len(documents)
        
    return 0

def clear_database():
    """Wipes the memory clean"""
    coll = get_mongo_collection()
    if coll is not None:
        coll.delete_many({})
        return True
    return False