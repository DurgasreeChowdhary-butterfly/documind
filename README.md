# 🧠 DocuMind AI — Production RAG PDF Chatbot

> Ask questions across multiple PDFs using natural language. Powered by LangChain, HuggingFace, Pinecone/FAISS, and Gemini.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 What is this?

DocuMind AI is a **production-grade Retrieval Augmented Generation (RAG)** application that lets you:

- Upload **multiple PDF documents**
- Ask **natural language questions** across all of them
- Get **accurate, grounded answers** from Gemini — with source attribution
- Never hallucinate — answers come strictly from your documents

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE                        │
│                                                              │
│  PDF Upload → PyMuPDF (extract) → LangChain (chunk)         │
│       → HuggingFace all-MiniLM-L6-v2 (embed)               │
│       → Pinecone / FAISS (store vectors)                    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                         QUERY PHASE                          │
│                                                              │
│  Question → HuggingFace (embed) → Pinecone/FAISS (search)   │
│       → LangChain RetrievalQA → Gemini 1.5 Flash            │
│       → Grounded Answer + Source Attribution                 │
└──────────────────────────────────────────────────────────────┘
```

---

## ⚡ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI + Uvicorn |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Text Chunking** | LangChain RecursiveCharacterTextSplitter |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | Pinecone (production) / FAISS (local fallback) |
| **Orchestration** | LangChain RetrievalQA |
| **LLM** | Google Gemini 1.5 Flash (or Pro) |
| **Frontend** | Vanilla JS + CSS (zero dependencies) |

---

## 📁 Project Structure

```
documind/
├── app/
│   ├── __init__.py
│   ├── config.py          # Central config — reads from .env
│   ├── loader.py          # PDF extraction, chunking, PDF registry
│   ├── vector_store.py    # Pinecone + FAISS abstraction layer
│   └── rag_chain.py       # LangChain RetrievalQA + Gemini
├── static/
│   └── index.html         # Full UI — dark professional theme
├── data/                  # PDFs + FAISS index stored here
├── main.py                # FastAPI app — all endpoints
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/documind-ai.git
cd documind-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional — leave empty to use FAISS locally
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=documind
```

**Get your keys:**
- Gemini: https://aistudio.google.com/app/apikey (free)
- Pinecone: https://app.pinecone.io (free tier available)

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

Open: **http://localhost:8000**

---

## 🔌 API Reference

### Health Check
```http
GET /api/health
```
```json
{
  "status": "healthy",
  "vector_store": "pinecone",
  "total_vectors": 432,
  "total_pdfs": 3,
  "model": "gemini-1.5-flash"
}
```

### Upload PDF
```http
POST /api/upload
Content-Type: multipart/form-data

file: <pdf_file>
```
```json
{
  "message": "'report.pdf' indexed successfully!",
  "filename": "report.pdf",
  "chunks": 47,
  "size_kb": 284.5,
  "vector_store": "pinecone"
}
```

### List PDFs
```http
GET /api/pdfs
```
```json
{
  "pdfs": [
    {
      "filename": "report.pdf",
      "num_chunks": 47,
      "file_size_kb": 284.5,
      "uploaded_at": "2025-01-15T10:30:00",
      "status": "indexed"
    }
  ],
  "total": 1
}
```

### Delete PDF
```http
DELETE /api/pdfs/{filename}
```

### Chat
```http
POST /api/chat
Content-Type: application/json

{
  "question": "What are the key findings?",
  "top_k": 4
}
```
```json
{
  "answer": "The key findings are...",
  "sources": ["report.pdf", "summary.pdf"],
  "model": "gemini-1.5-flash",
  "chunks_used": 4
}
```

---

## ⚙️ Configuration Options

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | required | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Use `gemini-1.5-pro` for higher quality |
| `PINECONE_API_KEY` | optional | Leave empty to auto-use FAISS |
| `PINECONE_INDEX_NAME` | `documind` | Pinecone index name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |
| `MAX_FILE_SIZE_MB` | `50` | Max PDF upload size |

---

## 🧠 How RAG Works (Quick Explainer)

```
1. UPLOAD:  PDF → extract text → split into 500-char chunks
2. EMBED:   Each chunk → 384-dim vector (HuggingFace model)
3. STORE:   Vectors saved to Pinecone or FAISS
4. QUERY:   Question → vector → find top 4 similar chunks
5. ANSWER:  Chunks + question → Gemini → grounded answer
```

The LLM never sees the full PDF — only the most relevant ~2000 characters per query. This makes it fast, cheap, and accurate.

---

## 🔄 Pinecone vs FAISS

| | Pinecone | FAISS |
|---|---|---|
| **Type** | Cloud (managed) | Local (in-memory) |
| **Setup** | Needs API key | Zero setup |
| **Scale** | Millions of vectors | Thousands of vectors |
| **Persistence** | Cloud — survives restarts | Disk — survives restarts |
| **Best for** | Production | Development / testing |

**Auto-selection:** If `PINECONE_API_KEY` is set → uses Pinecone. Otherwise → FAISS.

---

## 🐳 Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t documind-ai .
docker run -p 8000:8000 --env-file .env documind-ai
```

---

## 🛠️ Troubleshooting

**"Could not extract text from PDF"**
→ PDF is likely scanned/image-based. Use an OCR tool like Adobe Acrobat first.

**"No documents indexed yet"**
→ Upload a PDF via the sidebar before asking questions.

**Pinecone errors**
→ Check your API key in `.env`. Leave it empty to fall back to FAISS.

**Slow first response**
→ HuggingFace model downloads on first run (~90MB). Subsequent runs are instant.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built with Python, LangChain, and way too much coffee.
