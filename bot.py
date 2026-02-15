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
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Database Connected.")
except Exception as e:
    print(f"❌ Database Error: {e}")
    sys.exit(1)

# --- Search Function ---
async def search_knowledge_base(params):
    query = params.arguments.get("query")
    print(f"\n🔎 [RAG] Searching: '{query}'")
    try:
        vector = embedder.encode(query).tolist()
        results = collection.aggregate([
            {"$vectorSearch": {"index": "vector_index", "path": "embedding", "queryVector": vector, "numCandidates": 100, "limit": 1}}
        ])
        text_results = [doc['text'] for doc in results]
        if text_results:
            print(f"📄 Found info.") 
            await params.result_callback(f"Context: {text_results[0]}")
        else:
            print("❌ No info found.")
            await params.result_callback("No info found.")
    except Exception as e:
        await params.result_callback("Search failed.")

# --- Main Agent ---
async def main():
    print("\n--- 🎤 STARTING VOICE AGENT ---")
    print("1. Speak LOUDLY and CLEARLY.")
    print("2. Press Ctrl+C to stop.\n")

    tools = [{
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search for technical details.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }]

    # 2. Context (Memory)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a sophisticated Voice AI Agent designed by a developer."
                "You are built using Pipecat, Groq, Deepgram, and MongoDB."
                "IMPORTANT: If the user asks about 'tools', 'architecture', 'tech stack', or 'how you work', "
                "you MUST use the 'search_knowledge_base' tool to get the correct answer. "
                "Do not guess. Do not say you are a text interface."
                "Keep your responses short, conversational, and friendly."
            )
        }
    ]
    context = LLMContext(messages) 
    context_aggregator = LLMUserContextAggregator(context)

    # Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), model="nova-2")
    llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant", params=OpenAILLMService.InputParams(tools=tools))
    llm.register_function("search_knowledge_base", search_knowledge_base)
    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-asteria-en")

    # Transport
    # FIX: Use simple SileroVADAnalyzer without nested params
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer() 
        )
    )

    # Pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator,
        llm,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")