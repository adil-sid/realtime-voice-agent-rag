# Real-Time Voice Agent with RAG 🎙️🧠

An open-source implementation of a conversational AI agent featuring vector search (Retrieval-Augmented Generation) and sub-second voice response times. 

This project bridges the gap between natural voice interactions and highly accurate, context-aware AI responses, orchestrated using the **Pipecat** framework.

## 🌟 Overview

The **Real-Time Voice Agent** is designed to provide seamless, low-latency voice-to-voice interactions. By integrating a RAG backend, the agent doesn't just rely on the LLM's base knowledge—it retrieves relevant, domain-specific information from a vector database before speaking. This makes it perfect for customer support bots, interactive voice assistants, and enterprise knowledge retrieval systems where both speed and accuracy are critical.

## ✨ Key Features

* **Sub-Second Latency:** Highly optimized pipeline for Speech-to-Text (STT) and Text-to-Speech (TTS) to ensure natural conversational flow.
* **Retrieval-Augmented Generation (RAG):** Grounded responses using your own custom data, ensuring the agent provides factual and relevant answers.
* **Modular Pipeline:** Built on top of the **Pipecat** framework, allowing easy swapping of STT, LLM, and TTS providers.
* **Interactive UI:** Includes an out-of-the-box user interface (`agent_ui.py`) for easy testing and deployment.
* **Streamlined Ingestion:** Built-in scripts (`ingest.py`) to quickly chunk, embed, and index your documents into the vector database.

## 🛠️ Tech Stack

* **Core Language:** Python 3.10+
* **Framework Orchestration:** [Pipecat](https://github.com/pipecat-ai/pipecat)
* **Audio Processing:** State-of-the-art STT & TTS integration 
* **Knowledge Retrieval (RAG):** Custom Vector Database integration (via `rag_backend.py`)
* **User Interface:** Web-based UI (`agent_ui.py`)

## 🚀 Installation

**1. Clone the repository**
```bash
git clone [https://github.com/adil-sid/realtime-voice-agent-rag.git](https://github.com/adil-sid/realtime-voice-agent-rag.git)
cd realtime-voice-agent-rag
