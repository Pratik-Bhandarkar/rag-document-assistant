# 🗂️ RAG Document Assistant

A fully functional Retrieval-Augmented Generation (RAG) system built from scratch to query personal financial documents in natural language — supporting both English and German.

Built as a learning project and portfolio piece to demonstrate end-to-end AI engineering skills — from raw PDF extraction to a production-ready web application.

---

## 🎯 Project Overview

This system allows natural language querying of personal payslip documents, returning accurate, grounded answers with source citations. The assistant understands both English and German questions and responds in the same language as the query.

**Example queries:**
- "What was my net salary in November 2024?"
- "What was my health insurance contribution?"
- "Was ist mein Nettogehalt?"
- "How much income tax did I pay?"
- "Which bank was my salary transferred to?"
- "What was my gross salary?"

---

## 🏗️ Architecture

```
PDF Documents
     ↓
Text Extraction (PyMuPDF)
     ↓
Abbreviation Expansion (German payroll terms)
     ↓
Chunking with Overlap (1000 chars, 100 overlap)
     ↓
Multilingual Embeddings (sentence-transformers)
     ↓
Vector Storage with Metadata (ChromaDB)
     ↓
Semantic Search + Metadata Filtering
     ↓
LLM Answer Generation (Local or API)
     ↓
Streamlit Web UI with Citations
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| PDF Extraction | PyMuPDF (fitz) |
| Text Preprocessing | Custom abbreviation expansion with regex |
| Chunking | Custom Python pipeline with overlap |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Database | ChromaDB |
| LLM (Local) | Llama 3.2 via Ollama |
| LLM (API) | GPT-4o-mini via OpenAI API |
| UI | Streamlit |
| Language | Python 3.x |

---

## 📁 Project Structure

```
rag-document-assistant/
│
├── extract_text.py           # PDF text extraction pipeline
├── chunk_text.py             # Text chunking with abbreviation expansion and metadata
├── generate_embeddings.py    # Multilingual embedding generation
├── store_embeddings.py       # ChromaDB vector storage
├── search.py                 # Semantic search with metadata filtering
├── ask.py                    # Terminal RAG interface (local LLM)
├── app.py                    # Streamlit UI (local LLM via Ollama)
├── app_openai.py             # Streamlit UI (OpenAI API)
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Pratik-Bhandarkar/rag-document-assistant.git
cd rag-document-assistant
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install pymupdf sentence-transformers chromadb ollama streamlit openai python-dotenv
```

### 4. Add your documents
Create the following folder structure and name your payslip files using this convention:

```
documents/
└── payslips/
    └── Employer_Month_Year.pdf
```

Example:
```
documents/
└── payslips/
    ├── Employer1_Feb_2023.pdf
    ├── Employer2_Jan_2024.pdf
    └── Employer3_Sep_2025.pdf
```

### 5. Run the pipeline
```bash
python extract_text.py         # Step 1: Extract text from PDFs
python chunk_text.py           # Step 2: Expand abbreviations and chunk
python generate_embeddings.py  # Step 3: Generate embeddings
python store_embeddings.py     # Step 4: Store in ChromaDB
```

### 6. Run the app

**Local LLM version (requires Ollama):**
```bash
ollama pull llama3.2
streamlit run app.py
```

**OpenAI API version:**
```bash
# Create .env file with your API key
echo "OPENAI_API_KEY=your_key_here" > .env
streamlit run app_openai.py
```

---

## 🔍 Key Features

- **Multilingual support** — query in English or German, responses always match query language
- **Semantic search** — finds relevant content by meaning not just keywords
- **Metadata filtering** — filter by employer, month and year for precise retrieval
- **Dynamic retrieval** — fetches 5 chunks when filtered, 10 when unfiltered
- **Abbreviation expansion** — German payroll abbreviations expanded before embedding for better search quality
- **Source citations** — every answer shows exactly which document and chunk it came from
- **Two LLM backends** — local (private, free) and API (production quality)
- **Overlap chunking** — 100 character overlap preserves context at chunk boundaries

---

## 🔄 Two LLM Approaches

### Local (app.py)
- Uses Llama 3.2 3B via Ollama
- Completely private — no data leaves your machine
- Free to run
- Best for: privacy sensitive documents on capable hardware

### API (app_openai.py)
- Uses GPT-4o-mini via OpenAI API
- Superior instruction following and language detection
- Small cost per query (~$0.001)
- Best for: production quality answers on any hardware

---

## 🧠 Engineering Decisions

### Why overlap chunking?
Overlap chunking ensures that information at chunk boundaries is not lost. A 100 character overlap between consecutive chunks means the LLM always receives complete context even when relevant information spans a chunk boundary.

### Why dynamic retrieval?
When metadata filters are applied (employer + month + year), the search space is already narrow — 5 chunks is sufficient. Without filters, 10 chunks provides broader coverage for general questions.

### Why abbreviation expansion?
German payroll documents use domain specific abbreviations (KV-Beitrag, RV-Beitrag etc.) that embedding models may not recognise. Expanding these to full terms before embedding improves semantic search quality significantly. Regex word boundaries ensure only standalone abbreviations are expanded, preventing corruption of other text.

### Why two LLM backends?
Local LLMs offer complete privacy but are constrained by available VRAM. API based LLMs offer superior performance regardless of hardware. Providing both gives users a choice based on their privacy and quality requirements.

---

## ⚠️ Known Limitations

- **Aggregation queries** — questions requiring cross-document calculations ("total net earned in 2024", "highest salary across all employers") are not supported. This requires a structured data extraction layer beyond standard RAG.
- **Table structure preservation** — complex German payroll tables lose their column-value relationships during PDF text extraction. This affects some specific field lookups.
- **Local LLM language consistency** — the 3B parameter local model occasionally responds in the document language rather than the query language due to 4GB VRAM hardware constraints. The OpenAI API version handles this correctly.

---

## 🚀 Future Improvements

- [ ] Support for additional document types (tax statements, insurance letters, registration documents)
- [ ] Structured data extraction layer for aggregation queries
- [ ] LLM based metadata extraction for unstructured document collections
- [ ] Table aware PDF parsing to preserve payroll table structure
- [ ] Swap ChromaDB for Qdrant for production scalability
- [ ] AWS deployment pipeline
- [ ] LangGraph for multi-step reasoning and query routing
- [ ] Project 2: Hybrid RAG system with document intelligence for messy real world documents

---

## 💡 What I Learned

- End-to-end RAG pipeline architecture from scratch
- Semantic search with multilingual vector embeddings
- ChromaDB for vector storage and metadata filtering
- Prompt engineering for multilingual document QA
- System vs user prompt patterns in OpenAI API
- Tradeoffs between local LLMs and API based LLMs
- Chunk size and overlap tuning for document quality
- Regex based text preprocessing for domain specific documents
- Real world considerations for messy document collections
- Engineering judgement — knowing when to ship vs when to keep tuning

---

## 👨‍💻 Author

Pratik Bhandarkar — transitioning from Data Engineering to AI Engineering

[LinkedIn](https://linkedin.com/in/pratik-bhandarkar) | [GitHub](https://github.com/Pratik-Bhandarkar)