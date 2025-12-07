# 🔍 Knowledge Base Search Engine - RAG System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0A0A0A?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAWlBMVEUAAAD///////////////////////////////////////////////////////////////////////////////////////////////////+aL0CZAAAAG3RSTlMAAQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRo7KyYAAABHSURBVBjTZc7bEoAgDETRRkQCo9X9n5S3QJtBS9n/JyE0yoynZMhUlCT3VkOITkkpFmS6r30YIOCwVDDDeWGPAHDLcGRIDvDT+xr3aMcMBXNIN1qQEbfQAAAABJRU5ErkJggg==&labelColor=green)
![Groq](https://img.shields.io/badge/Groq-FF6B00?logo=lightning&logoColor=white)

A production-ready Retrieval-Augmented Generation (RAG) system for document search and question answering using FastAPI, ChromaDB, Groq, and Streamlit.

## 🎯 Features

- **Multi-Document Support**: Upload and process multiple PDF and TXT files
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Context-Aware Answers**: LLM-powered synthesis with Groq (llama-3.3-70b-versatile)
- **Source Citations**: Tracks which documents were used for answers
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Interactive UI**: Streamlit frontend for easy interaction
- **Persistent Storage**: ChromaDB for vector embeddings

## 🏗️ Architecture

```
User Query → API → Vector Retrieval (ChromaDB) → Top K Chunks → LLM (Groq) → Synthesized Answer
```

## 📋 Prerequisites

- Python 3.10 or higher
- Groq API key

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ShreyasBorade30/knowledge-base-search-engine.git
cd knowledge-base-rag
```

### 2. Install Dependencies with uv (Fast!)

```bash
# pip install uv(if uv is not installed)

# Install dependencies
uv pip install -r backend/requirements.txt
```

**Alternative: Using standard pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file inside the backend folder:

```bash
cp backend/.env.example backend/.env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
API_URL=http://localhost:8000
```


## 🎮 Usage

### Option 1: Using uv (Recommended - Faster)

#### Terminal 1 - Start Backend API

```bash
uv run python backend/main.py
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

#### Terminal 2 - Start Streamlit UI

```bash
uv run streamlit run frontend/streamlit_app.py
```

The UI will open automatically at `http://localhost:8501`

### Option 2: Using standard Python

#### Terminal 1 - Start Backend API

```bash
cd backend
python main.py
```

#### Terminal 2 - Start Streamlit UI

```bash
cd frontend
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
knowledge-base-rag/
├── backend/
│ ├── main.py # FastAPI backend (upload, query, stats)
│ ├── rag_engine.py # RAG logic: extraction, chunking, embeddings, ChromaDB
│ ├── requirements.txt # Backend dependencies
│ ├── data/
│ │ └── documents/ # Uploaded documents (auto-created, ignored by Git)
│ ├── chroma_db/ # ChromaDB persistent storage (auto-created after first upload)
│ └── __pycache__/ # Python cache files
│
├── frontend/
│ └── streamlit_app.py # Streamlit UI
│
├── .venv/ # Virtual environment (ignored)
├── .gitignore # Git ignore rules
├── .python-version # Python version indicator
├── pyproject.toml # Project metadata
├── uv.lock # Lock file
└── README.md # Project documentation
└── kb-search_engine-Demo # Demo video

```

## 🔧 Technical Stack

### Backend
- **FastAPI**: REST API framework
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Text embeddings (all-MiniLM-L6-v2)
- **Groq**: LLM inference (llama-3.3-70b-versatile)
- **PyPDF2**: PDF text extraction

### Frontend
- **Streamlit**: Web interface
- **Requests**: API communication

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| POST | `/upload` | Upload and ingest documents |
| POST | `/query` | Query knowledge base |
| GET | `/stats` | Get system statistics |
| DELETE | `/clear` | Clear knowledge base |
| GET | `/health` | Health check |


## ⚙️ Configuration Options

### RAG Engine Parameters

In `rag_engine.py`, you can customize:

- **Chunk Size**: `chunk_size=500` (words per chunk)
- **Overlap**: `overlap=50` (overlap between chunks)
- **Top K**: `top_k=5` (number of chunks to retrieve)
- **Embedding Model**: `all-MiniLM-L6-v2` (can use other sentence-transformers)
- **LLM Model**: `llama-3.3-70b-versatile` (Groq model)
- **Temperature**: `temperature=0.3` (LLM creativity)


## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution**: Ensure `.env` file exists with your API key

### Issue: "No documents in knowledge base"
**Solution**: Upload documents first before querying

### Issue: "API connection failed"
**Solution**: Ensure backend is running on port 8000

### Issue: ChromaDB errors
**Solution**: Delete `backend/chroma_db` folder and restart
  

## 👨‍💻 Author

Shreyash Borade
shreyashborade04@gmail.com  

