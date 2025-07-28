#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Challenge 1B
Advanced Persona-Based Content Analysis

This solution builds upon Challenge 1A by implementing a Retrieval Augmented Generation (RAG) system
for advanced persona-based content analysis across multiple document collections. Key features include:
1. Semantic search and retrieval using Sentence Transformers and ChromaDB.
2. Persona-based summarization and question answering using Ollama (local LLM).
3. Scalable processing of multiple PDF documents.
4. Offline operation with pre-loaded models.
"""

import os
import sys
import json
from pathlib import Path
import time
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from challenge_1b import convert_pdf_to_html_and_extract_headings, extract_pdf_content_by_pages

# Configuration
CHROMA_DB_PATH = "/app/chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # This model needs to be pre-downloaded or available offline
OLLAMA_MODEL_NAME = "llama2" # This model needs to be pre-downloaded or available offline

class DocumentProcessor:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name="pdf_documents")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def process_and_embed_pdf(self, pdf_path: Path, output_dir: Path):
        print(f"Processing and embedding: {pdf_path.name}")
        
        # 1. Extract headings and content
        headings_data = convert_pdf_to_html_and_extract_headings(str(pdf_path), str(output_dir))
        pdf_content_by_page = extract_pdf_content_by_pages(str(pdf_path))
        
        if not headings_data or not pdf_content_by_page:
            print(f"Skipping {pdf_path.name} due to extraction failure.")
            return

        document_id = pdf_path.stem
        
        # Prepare documents for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        # Add full page content as documents
        for page_num, content in pdf_content_by_page.items():
            documents.append(content)
            metadatas.append({"source": pdf_path.name, "page": page_num, "type": "page_content"})
            ids.append(f"{document_id}_page_{page_num}")
            
        # Add headings as separate, searchable documents
        for heading in headings_data.get("outline", []):
            documents.append(heading["text"])
            metadatas.append({"source": pdf_path.name, "page": heading["page"], "type": "heading", "level": heading["level"]})
            ids.append(f"{document_id}_heading_{heading["level"]}_{heading["page"]}_{heading["text"][:20].replace(' ', '_')}")

        # Generate embeddings and add to ChromaDB
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully embedded {len(documents)} chunks from {pdf_path.name}")

    def query_documents(self, query_text: str, persona: str = "general", n_results: int = 5):
        print(f"Querying documents for: '{query_text}' with persona '{persona}'")
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=
            ["documents", "metadatas"]
        )
        return results

    def generate_response_with_ollama(self, query: str, context: List[str], persona: str):
        print(f"Generating response with Ollama for persona: {persona}")
        
        # Construct a persona-based prompt
        if persona == "technical_analyst":
            system_prompt = "You are a technical analyst. Provide concise, factual, and highly technical answers based on the provided context. Focus on data, processes, and technical implications."
        elif persona == "executive_summary":
            system_prompt = "You are an executive assistant. Summarize the key takeaways and business implications from the provided context. Be brief and to the point."
        elif persona == "legal_expert":
            system_prompt = "You are a legal expert. Analyze the provided text for legal implications, compliance, and potential risks. Cite relevant sections if possible."
        else:
            system_prompt = "You are a helpful assistant. Answer the question based on the provided context."
            
        full_context = "\n\n".join(context)
        
        prompt = f"Context: {full_context}\n\nQuestion (for {persona}): {query}\n\nAnswer:"
        
        try:
            response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            return response["message"]["content"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "Could not generate response." # Fallback

def main():
    print("Adobe India Hackathon 2025 - Challenge 1B")
    print("Advanced Persona-Based Content Analysis")
    print("=" * 50)
    
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = DocumentProcessor()
    
    # Phase 1: Process and embed all PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in /app/input directory for embedding.")
    else:
        for pdf_file in pdf_files:
            processor.process_and_embed_pdf(pdf_file, output_dir)
            
    print("\nEmbedding phase complete. Ready for querying.")
    
    # Phase 2: Example Querying (This part would typically be driven by external input)
    # For hackathon submission, we can demonstrate a few hardcoded queries
    
    example_queries = [
        {"query": "What are the main components of the system architecture?", "persona": "technical_analyst"},
        {"query": "Provide a high-level summary of the document's purpose.", "persona": "executive_summary"},
        {"query": "Are there any mentions of data privacy regulations?", "persona": "legal_expert"},
        {"query": "What is the overall theme of the hackathon?", "persona": "general"}
    ]
    
    results_list = []
    for example in example_queries:
        query = example["query"]
        persona = example["persona"]
        
        retrieved_results = processor.query_documents(query, persona=persona, n_results=3)
        
        context_for_llm = [doc for doc in retrieved_results["documents"][0]] # Assuming documents are in the first list
        
        ollama_response = processor.generate_response_with_ollama(query, context_for_llm, persona)
        
        results_list.append({
            "query": query,
            "persona": persona,
            "retrieved_documents": retrieved_results,
            "ollama_response": ollama_response
        })
        
        print(f"\n--- Query: {query} (Persona: {persona}) ---")
        print(f"Ollama Response: {ollama_response}")
        print("--------------------------------------------------")
        
    # Save results to a JSON file
    with open(output_dir / "query_results.json", "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"\nQuery results saved to {output_dir / 'query_results.json'}")

if __name__ == "__main__":
    main()

