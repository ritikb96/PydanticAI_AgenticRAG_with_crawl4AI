# Pydantic AI: Documentation Crawler & RAG Agent 🌟

An intelligent documentation crawler and Retrieval-Augmented Generation (RAG) agent built with **Pydantic**, **CRAWL4AI**, **Supabase**, and **OpenAI**. This system crawls documentation websites, stores content in a vector database, and provides precise, context-aware answers to user queries by retrieving and analyzing relevant documentation chunks. Designed for developers, data scientists, and tech enthusiasts, it streamlines workflows and transforms documentation into a dynamic knowledge resource. 🚀

## Features ✨

- 📚 **Documentation Website Crawling & Chunking**: Systematically crawls websites and splits content into meaningful chunks.
- 🗄️ **Vector Database Storage**: Stores chunks and metadata in Supabase with vector similarity search capabilities.
- 🔍 **Semantic Search**: Leverages OpenAI embeddings for accurate, context-driven query matching.
- 💬 **RAG-Based Question Answering**: Combines retrieval with generative models (e.g., gpt-4o-mini) for insightful responses.
- 💾 **Code Block Preservation**: Ensures technical accuracy by preserving code blocks.
- 🌐 **Streamlit UI**: Provides an interactive web interface for querying.
- ⚙️ **API Endpoint**: Supports programmatic access for custom integrations.

## Prerequisites 📋

- Python 3.11+
- Supabase account and database
- OpenAI API key
- Streamlit (for web interface)

## Installation 🛠️

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ritikb96/PydanticAI_AgenticRAG_with_crawl4AI
  
   ```

2. **Install Dependencies** (recommended to use a Python virtual environment):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Rename `.env.example` to `.env`.
   - Edit `.env` with your API keys and preferences:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     SUPABASE_URL=your_supabase_url
     SUPABASE_SERVICE_KEY=your_supabase_service_key
     LLM_MODEL=gpt-4o-mini  # or your preferred OpenAI model
     ```

## Usage 🚀

### Database Setup 🗄️

Execute the SQL commands in `site_pages.sql` to:
- Create necessary tables
- Enable vector similarity search
- Set up Row Level Security policies

In Supabase:
1. Go to the "SQL Editor" tab.
2. Paste the SQL from `site_pages.sql`.
3. Click "Run".

### Crawl Documentation 📚

To crawl and store documentation in the vector database:
```bash
python crawl_pydantic_ai_docs.py
```

This will:
- Fetch URLs from the documentation sitemap
- Crawl each page and split into chunks
- Generate embeddings and store in Supabase

### Streamlit Web Interface 🌐

For an interactive web interface to query the documentation:
```bash
streamlit run streamlit_ui.py
```

Access the interface at `http://localhost:8501`.

## Configuration ⚙️

### Database Schema

The Supabase database uses the following schema:
```sql
CREATE TABLE site_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT,
    chunk_number INTEGER,
    title TEXT,
    summary TEXT,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);
```

### Chunking Configuration

Configure chunking parameters in `crawl_pydantic_ai_docs.py`:
```python
chunk_size = 5000  # Characters per chunk
```

The chunker preserves:
- Code blocks
- Paragraph boundaries
- Sentence boundaries

## Project Structure 📂

- `crawl_pydantic_ai_docs.py`: Documentation crawler and processor
- `pydantic_ai_expert.py`: RAG agent implementation
- `streamlit_ui.py`: Web interface
- `site_pages.sql`: Database setup commands
- `requirements.txt`: Project dependencies

## Why This Project Matters 🌍

This project addresses the challenge of navigating complex documentation by automating content extraction, indexing, and querying. By combining web crawling, semantic search, and RAG, it empowers users to access relevant information quickly, enhancing developer productivity, technical support, education, and enterprise knowledge management. Its modular design makes it a strong foundation for further innovation in AI-driven workflows.

## Credit 🙌

This project was inspired by an insightful YouTube tutorial. Learn more about it on the creator’s channel: https://www.youtube.com/@ColeMedin

## Contributing 🤝

Contributions are welcome! Please open an issue or submit a pull request to share improvements or ideas.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

