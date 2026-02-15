import streamlit as st
import subprocess
import os
import signal
import sys
import rag_backend

# Page Config
st.set_page_config(page_title="AI Voice Agent Control", page_icon="🎙️", layout="wide")

st.title("🎙️ AI Voice Agent Control Center")
st.markdown("Upload documents, manage memory, and start your Voice Agent.")

# --- Session State (Keeps track of the running agent) ---
if 'bot_process' not in st.session_state:
    st.session_state.bot_process = None

# --- SIDEBAR: RAG Controls ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.info("Upload a PDF (Resume, Report, etc.) to teach the agent new facts.")
    
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("🚀 Process & Ingest"):
            with st.spinner("Reading and memorizing..."):
                try:
                    # Call the backend function
                    num_chunks = rag_backend.process_and_store_pdf(uploaded_file)
                    st.success(f"✅ Learned {num_chunks} new facts from '{uploaded_file.name}'!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.subheader("Memory Management")
    if st.button("🗑️ Wipe Memory"):
        if rag_backend.clear_database():
            st.warning("⚠️ Memory wiped clean!")
        else:
            st.error("Could not connect to database.")

# --- MAIN AREA: Agent Controls ---
st.subheader("🤖 Live Agent Control")

col1, col2 = st.columns(2)

with col1:
    # START BUTTON
    if st.button("▶️ START VOICE AGENT", type="primary", use_container_width=True):
        if st.session_state.bot_process is None:
            try:
                # 1. Get the path to the current Python (inside voiceenv)
                python_executable = sys.executable 
                
                # 2. Command: Open CMD, Keep it open (/k), and run the script
                # This ensures you can see any errors if it crashes.
                cmd_command = [
                    "cmd.exe", 
                    "/k", 
                    python_executable, 
                    "voice_bot.py"
                ]
                
                st.session_state.bot_process = subprocess.Popen(
                    cmd_command, 
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                st.success("Agent is starting... Check the new terminal window!")
            except Exception as e:
                st.error(f"Failed to start: {e}")

with col2:
    # STOP BUTTON
    if st.button("🛑 STOP AGENT", type="secondary", use_container_width=True):
        if st.session_state.bot_process is not None:
            # Kill the process forcefully
            st.session_state.bot_process.terminate()
            st.session_state.bot_process = None
            st.error("Agent Stopped.")
        else:
            st.info("Agent is not running.")

# --- Status Indicator ---
st.divider()
if st.session_state.bot_process:
    st.success("🟢 STATUS: Agent is RUNNING")
else:
    st.error("🔴 STATUS: Agent is STOPPED")

st.markdown("""
---
**Instructions:**
1. Upload a PDF in the sidebar to add knowledge.
2. Click **Start Voice Agent**.
3. A black terminal window will appear.
4. Speak to your agent! It now knows what is in your PDF.
""")