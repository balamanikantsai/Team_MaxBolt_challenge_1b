# Adobe India Hackathon 2025 - Challenge 1B

## Advanced Persona-Based Content Analysis

### Overview

This solution extends the capabilities of Challenge 1A by implementing a sophisticated Retrieval Augmented Generation (RAG) system. It enables advanced persona-based content analysis across multiple PDF documents, providing intelligent answers to complex queries.

### Key Features

1. **Semantic Search & Retrieval**: Utilizes Sentence Transformers to create embeddings of document content and ChromaDB as a vector store for efficient semantic search.

2. **Persona-Based Question Answering**: Integrates with Ollama (a local LLM) to generate contextually relevant answers tailored to specific personas (e.g., Technical Analyst, Executive Summary, Legal Expert).

3. **Multi-Document Processing**: Capable of processing and analyzing multiple PDF documents, building a comprehensive knowledge base.

4. **Offline Operation**: Designed to work entirely offline, with pre-downloaded models for embeddings and the LLM, adhering to hackathon constraints.

### Libraries Used

- **docling**: PDF to HTML conversion
- **beautifulsoup4**: HTML parsing
- **langchain-community**: PDF content extraction
- **pypdf**: Fallback PDF processing
- **numpy**: Numerical operations
- **chromadb**: Vector database for storing and querying document embeddings
- **sentence-transformers**: For generating semantic embeddings of text
- **ollama**: Local Large Language Model (LLM) for generative AI tasks

### Architecture

```
PDF Inputs → (Docling, BeautifulSoup, PyPDF) → Text Chunks & Headings
                               ↓
                         Sentence Transformers → Embeddings
                                               ↓
                                            ChromaDB (Vector Store)
                                               ↓
Query & Persona → Semantic Search (ChromaDB) → Retrieved Context
                                               ↓
                                            Ollama (LLM) → Persona-Based Answer
```

1. **Document Ingestion**: PDFs are processed to extract text content and headings. These are then chunked and converted into numerical embeddings.
2. **Vector Storage**: Embeddings are stored in ChromaDB, allowing for fast similarity searches.
3. **Query Processing**: User queries are embedded and used to retrieve relevant document chunks from ChromaDB.
4. **Generative AI**: The retrieved context, along with the user's query and specified persona, is fed into Ollama to generate a tailored response.

### Innovation Points

- **Full RAG Implementation**: Demonstrates a complete RAG pipeline for enhanced question answering beyond simple keyword search.
- **Persona-Driven Responses**: Provides a unique capability to generate answers from different perspectives, adding significant value for diverse users.
- **Local LLM Integration**: Utilizes Ollama for on-device AI, ensuring privacy, speed, and compliance with offline requirements.
- **Scalable Knowledge Base**: ChromaDB allows for easy expansion to large document collections.

### How to Build and Run

```bash
# Build the Docker image
docker build --platform linux/amd64 -t challenge1b:latest .

# Run the solution
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1b:latest
```

### Input/Output Format

**Input**: PDF files in `/app/input/` directory

**Output**: JSON file in `/app/output/` containing query results and Ollama responses.

### Performance Characteristics

- **Speed**: Optimized for efficient embedding generation and retrieval.
- **Accuracy**: High relevance of retrieved information due to semantic search.
- **Memory**: Manages embeddings and LLM efficiently within resource constraints.
- **CPU**: Designed for CPU-only operation.

### Pre-requisites for Ollama and Sentence Transformers

For a fully functional offline setup, the `all-MiniLM-L6-v2` Sentence Transformer model and the `llama2` Ollama model need to be pre-downloaded. In a real deployment, these would be part of the Docker image build process or mounted volumes.

**Note**: The Dockerfile includes commands to install Ollama and attempt to download the Sentence Transformer model. For the Ollama model, you would typically run `ollama pull llama2` before building the final image or have it available in the environment where the Docker container runs.

