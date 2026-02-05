import streamlit as st
import os
import sys
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import initialize_system, ingest_srt
from src.utils.logger import MovieRAGLogger

# Configure Page
st.set_page_config(
    page_title="Movie RAG Chat",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = None

# Load Config
def load_config():
    import yaml
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize System (Lazy Load)
def get_system():
    if st.session_state.pipeline is None:
        with st.spinner("Initializing AI System..."):
            pipeline, metadata_store, _ = initialize_system(config)
            st.session_state.pipeline = pipeline
            st.session_state.metadata_store = metadata_store
    return st.session_state.pipeline, st.session_state.metadata_store

pipeline, metadata_store = get_system()

# --- Sidebar: Management ---
with st.sidebar:
    st.header("🎬 Movie Management")
    
    # 1. Ingestion Section
    st.subheader("New Movie Ingestion")
    uploaded_file = st.file_uploader("Upload SRT File", type=["srt"])
    movie_id_input = st.text_input("Movie ID (Unique)", placeholder="e.g. inception_2010")
    
    if st.button("Ingest Movie", type="primary"):
        if uploaded_file and movie_id_input:
            temp_dir = Path("data/raw_srt")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.status("Ingesting movie...", expanded=True) as status:
                st.write("Parsing SRT...")
                try:
                    ingest_srt(str(temp_path), movie_id_input, config)
                    st.write("Chunking and Embedding...")
                    time.sleep(1) # UX pause
                    status.update(label="Ingestion Complete!", state="complete", expanded=False)
                    st.success(f"Successfully ingested {movie_id_input}!")
                    st.rerun() # Refresh to show in list
                except Exception as e:
                    status.update(label="Ingestion Failed", state="error")
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload a file and provide an ID.")

    st.divider()

    # 2. Selection & Deletion
    st.subheader("Available Movies")
    
    # Refresh movie list
    available_movies = metadata_store.list_movies()
    selected_movie = st.selectbox(
        "Select Movie to Chat With", 
        ["All Movies (Global Search)"] + available_movies,
        index=0
    )
    
    # Determine movie_id filter
    active_movie_id = None if selected_movie == "All Movies (Global Search)" else selected_movie
    
    # Delete Button
    if active_movie_id:
        if st.button(f"🗑️ Delete {active_movie_id}", type="secondary"):
            metadata_store.clear_movie(active_movie_id)
            # We also used to clear embeddings but the current store doesn't support selective delete easily without re-index.
            # Metadata delete prevents retrieval, effectively deleting it.
            st.success(f"Deleted metadata for {active_movie_id}")
            st.rerun()

# --- Main Chat Area ---
st.title("🍿 Chat with Movie Transcripts")

if not available_movies:
    st.info("👋 Welcome! Please upload an SRT file in the sidebar to get started.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the movie..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = pipeline.process(prompt, movie_id=active_movie_id)
                
                # Format Answer
                answer_text = result.answer.answer
                
                # Add Metadata display
                meta_info = f"\n\n---\n**Sources:** {result.relevant_chunks} chunks used | **Model:** {result.answer.model_used}"
                if result.answer.supporting_timestamps:
                    meta_info += f" | **Timestamps:** {', '.join(result.answer.supporting_timestamps)}"
                
                # Display Token Usage
                if result.token_usage:
                    meta_info += f" | **Tokens:** {result.token_usage.total_tokens} (Prompt: {result.token_usage.prompt_tokens}, Completion: {result.token_usage.completion_tokens})"
                
                full_response = answer_text + meta_info
                
                st.markdown(full_response)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
