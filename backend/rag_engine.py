import os
from typing import List, Dict
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
import PyPDF2
from pathlib import Path

class RAGEngine:
    def __init__(self, groq_api_key: str, persist_directory: str = "./chroma_db"):
        """Initialize RAG engine with ChromaDB and Groq"""
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def ingest_document(self, file_path: str, document_name: str = None) -> Dict:
        """Process and store document in ChromaDB"""
        try:
            file_path = Path(file_path)
            if document_name is None:
                document_name = file_path.name
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                text = self.extract_text_from_txt(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Create unique IDs for chunks
            chunk_ids = [f"{document_name}_chunk_{i}" for i in range(len(chunks))]
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=chunk_ids,
                metadatas=[{"source": document_name, "chunk_id": i} for i in range(len(chunks))]
            )
            
            return {
                "status": "success",
                "document_name": document_name,
                "chunks_created": len(chunks),
                "message": f"Successfully ingested {document_name}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant chunks from ChromaDB"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        context_chunks = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                context_chunks.append({
                    "text": doc,
                    "source": results['metadatas'][0][i]['source'],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return context_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate answer using Groq LLM"""
        # Build context from retrieved chunks
        context = "\n\n".join([f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use the context below to answer the user's question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer the question succinctly and cite the sources used."""
        
        try:
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable assistant that provides accurate answers based on given context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = chat_completion.choices[0].message.content
            
            return {
                "status": "success",
                "answer": answer,
                "sources": list(set([chunk['source'] for chunk in context_chunks])),
                "context_used": len(context_chunks)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating answer: {str(e)}"
            }
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Main query method - retrieve and generate answer"""
        # Check if collection has documents
        if self.collection.count() == 0:
            return {
                "status": "error",
                "message": "No documents in knowledge base. Please upload documents first."
            }
        
        # Retrieve relevant context
        context_chunks = self.retrieve_context(question, top_k)
        
        if not context_chunks:
            return {
                "status": "error",
                "message": "No relevant information found in the knowledge base."
            }
        
        # Generate answer
        result = self.generate_answer(question, context_chunks)
        result['retrieved_chunks'] = context_chunks
        
        return result
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        try:
            count = self.collection.count()
            return {
                "status": "success",
                "total_chunks": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base"""
        try:
            self.chroma_client.delete_collection("knowledge_base")
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            return {"status": "success", "message": "Knowledge base cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}