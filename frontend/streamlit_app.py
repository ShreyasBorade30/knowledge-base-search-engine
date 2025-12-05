import streamlit as st
import requests
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Knowledge Base Search Engine",
    page_icon="🔍",
    layout="wide"
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-text {
        font-size: 14px;
        color: #666;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 Knowledge Base Search Engine")
st.markdown("Upload documents and ask questions using AI-powered RAG")

# Sidebar
with st.sidebar:
    st.header("📊 System Info")
    
    # Get stats
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        else:
            st.warning("Could not fetch stats")
    except:
        st.error("API connection failed")
    
    st.divider()
    
    # Clear knowledge base
    st.header("🗑️ Management")
    if st.button("Clear Knowledge Base", type="secondary"):
        try:
            response = requests.delete(f"{API_URL}/clear")
            if response.status_code == 200:
                st.success("Knowledge base cleared!")
                st.rerun()
            else:
                st.error("Failed to clear knowledge base")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.divider()
    st.markdown("""
    ### About
    This app uses:
    - **FastAPI** for backend
    - **ChromaDB** for vector storage
    - **Groq** for LLM inference
    - **Streamlit** for UI
    """)

# Main content tabs
tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload Documents")
    st.markdown("Upload PDF or TXT files to add them to the knowledge base")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT documents"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Upload file to API
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {uploaded_file.name}: {result.get('chunks_created', 0)} chunks created")
                    else:
                        st.error(f"❌ {uploaded_file.name}: Upload failed")
                        
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            st.toast("🎉 File processed successfully!")


# Tab 2: Query Interface
with tab2:
    st.header("Ask Questions")
    st.markdown("Query your knowledge base using natural language")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Your question",
            placeholder="e.g., What are the main findings in the research papers?",
            label_visibility="collapsed"
        )
    
    with col2:
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=5, help="Number of chunks to retrieve")
    
    if st.button("🔍 Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching knowledge base..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": query, "top_k": top_k}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result["status"] == "success":
                            # Display answer
                            st.subheader("💡 Answer")
                            st.markdown(result["answer"])
                            
                            # Display sources
                            st.subheader("📚 Sources")
                            sources = result.get("sources", [])
                            if sources:
                                for source in sources:
                                    st.markdown(f"- {source}")
                            
                            # Display metadata
                            with st.expander("🔧 Query Details"):
                                st.write(f"**Chunks Retrieved:** {result.get('context_used', 0)}")
                                st.write(f"**Top K Parameter:** {top_k}")
                        else:
                            st.error(result.get("message", "Query failed"))
                    else:
                        st.error("API request failed")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What are the main topics discussed in the documents?
        - Summarize the key findings from the research
        - What recommendations are mentioned?
        - Compare the different approaches discussed
        - What are the limitations mentioned in the papers?
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Built with FastAPI, ChromaDB, Groq, and Streamlit | RAG-powered Knowledge Base</small>
</div>
""", unsafe_allow_html=True)

