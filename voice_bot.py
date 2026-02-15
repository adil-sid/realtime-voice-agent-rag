import asyncio
import os
import sys
import pymongo
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer

# --- Pipecat Imports ---
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import TextFrame

# Processors
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator
from pipecat.processors.aggregators.llm_context import LLMContext

# Services
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService 
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

load_dotenv()

# --- Database Setup ---
try:
    mongo_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
    collection = mongo_client["voice_agent_db"]["knowledge_base"]
    
    print("🧠 Loading AI Model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    print("✅ Database & AI Model Ready.")
except Exception as e:
    print(f"❌ Database Error: {e}")
    sys.exit(1)

# --- Search Function (The "RAG" Tool) ---
async def search_knowledge_base(params):
    query = params.arguments.get("query")
    print(f"\n🔎 [RAG] Searching for: '{query}'...") 
    
    try:
        # Convert user query to vector
        vector = embedder.encode(query).tolist()
        
        # Search MongoDB
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": 100,
                    "limit": 3  # Increased limit to 3 chunks for better summaries
                }
            }
        ])
        
        text_results = [doc['text'] for doc in results]
        
        if text_results:
            print(f"📄 Found {len(text_results)} matches.") 
            combined_text = "\n\n".join(text_results)
            await params.result_callback(f"Here is the content found in the database:\n{combined_text}")
        else:
            print("❌ No matches found in PDF.")
            # Fallback: If vector search fails, tell the LLM to just guess or ask for specifics
            await params.result_callback("I searched the database but found no exact matches. Ask the user for a specific topic.")

    except Exception as e:
        print(f"❌ Search Error: {e}")
        await params.result_callback("Database search failed.")

# --- Main Agent ---
async def main():
    print("\n--- 🎤 VOICE AGENT READY ---")
    print("1. Speak clearly.")
    print("2. Ask: 'Summarize the document' or 'What is this about?'")
    print("3. Press Ctrl+C to stop.\n")

    # 1. Define the Tool
    tools = [{
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "MANDATORY: Use this tool to find information. If user asks for 'summary', search for 'introduction' or 'main points'.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search keywords. For summaries, use 'overview' or 'abstract'."
                    }
                }, 
                "required": ["query"]
            }
        }
    }]

    # 2. System Prompt (The Personality)
    # This is the "Brainwash" part
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent Voice AI connected to a Knowledge Base."
                "CONTEXT: The user has ALREADY uploaded a PDF document to your database."
                "INSTRUCTION: When the user asks 'Summarize this' or 'What is this about?', you MUST use the 'search_knowledge_base' tool."
                "Do NOT say 'I don't have a document'. instead, say 'Let me check the database...' and use the tool."
                "Keep answers short and conversational."
            )
        }
    ]
    
    context = LLMContext(messages) 
    context_aggregator = LLMUserContextAggregator(context)

    # 3. Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), model="nova-2")
    
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant", # Using the NEW Model
        params=OpenAILLMService.InputParams(tools=tools)
    )
    llm.register_function("search_knowledge_base", search_knowledge_base)

    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-asteria-en")

    # 4. Transport (Speaker/Mic)
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer() 
        )
    )

    # 5. Pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator,
        llm,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False))
    
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")