<!-- ------------------------------------------------------------------- -->
<!--    DocuChat RAG Application – README                                    -->
<!-- ------------------------------------------------------------------- -->

<h1 align="center">DocuChat RAG Application 🤖📚</h1>
<h4 align="center">AI-powered support chat agents for multiple companies, backed by PDFs + WordPress integration</h4>

<p align="center">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110+-green">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

---

## ✨ What is this?

**DocuChat** is a **FastAPI backend** that powers customer-support chatbot agents for different companies (e.g., DigiRoam, DigiCom, DigiTech).  

- Upload company PDFs 📂  
- Store & index embeddings in **Pinecone**  
- Query with **RAG (Retrieval-Augmented Generation)** via **OpenAI GPT-4o-mini**  
- Maintain **session-based memory** in **Redis**  
- Connect seamlessly with **WordPress websites** via plugin  

|                      |                                                                                  |
|----------------------|----------------------------------------------------------------------------------|
| **Use case**         | Support chatbot agent linked to each WordPress site                     |
| **Tech stack**       | FastAPI · LangChain · OpenAI · Pinecone · Redis · pdfplumber                     |
| **Status**           | **Stable v1.0** – optimized for production deployment on Linux servers           |

---

## 🚀 Key Features

| Feature | Details |
|---------|---------|
| 🏢 **Multi-Tenant Isolation** | Each company uses its own Pinecone namespace (e.g. `tenant_digicom`, `tenant_digitech`) |
| 📄 **PDF Uploads** | Upload any PDF (validated for size & length), automatically chunked & embedded |
| 🔎 **Hybrid Retrieval** | Uses **Pinecone vectors + BM25 keyword retriever** for more accurate context |
| 🧠 **Session Memory** | Maintains conversation history per session (via Redis) with auto-expiry (e.g. 5 mins) |
| 💬 **Context-Grounded Answers** | GPT-4o-mini responds **only from uploaded PDFs**, no hallucinations |
| 🔐 **Admin Tools** | Endpoints to delete vectors by `pdf_hash` or entire `company_id` |
| 🌐 **WordPress Ready** | Easily integrated into WordPress websites via REST plugin |

---

## ⚙️ Quick Start (Run Locally)

### 🧑‍💻 1. Clone the Repository

```bash
git clone https://github.com/KhurramShams/DocuChat-RAG-Application-FastAPI.git
cd DocuChat-RAG-Application-FastAPI
````

---

### 📦 2. Create a Virtual Environment

```bash
python -m venv venv
# Activate it:
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

---

### 📂 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔐 4. Add Your API Keys

Create a `.env` file in the root directory of the project and add:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=pcd-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
DOCS_USERNAME = "...."  # use for Doc end-point access
DOCS_PASSWORD = "...."   # use for Doc end-point access
REDIS_URL=redis-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx    # or your Redis Cloud API
```

⚠️ Make sure `.env` is listed in `.gitignore` to avoid committing sensitive keys.

---

### 🚀 5. Run the FastAPI App

```bash
uvicorn main:app --reload
```

The app will open at:

```
http://127.0.0.1:8000
```

Docs (protected with Basic Auth) at:

```
http://127.0.0.1:8000/api-endpoint-links
```

---

## 📄 API Endpoints

### 📥 Upload PDF

`POST /upload_pdf/`

Form data:

```json
{
  "file": "<PDF file>",
  "company_id": "Domain-Name"
}
```

Response:

```json
{
  "stored": true,
  "hash": "abcdef123456...",
  "msg": "Stored embeddings in Pinecone.",
  "chunks": 42
}
```

---

### 💬 Ask a Question

`POST /ask/`

```json
{
  "company_id": "Domain-Name",
  "question": "What services does DigiTech provide?",
  "session_id": "user12345",
  "hash": ""
}
```

Response:

```json
{
  "answer": "DigiTech provides Social Media Marketing, Digital Advertising, SEO, Web Development & Design, and Digital PR services."
}
```

---

### ❌ Delete Vectors

Delete by **hash** or entire **company**.

`DELETE /delete_pdf_vectors/`

```json
# Option 1: delete by hash
{
  "hash": "abcdef123456"
}

# Option 2: delete all PDFs for a company
{
  "company_id": "Domain-Name"
}
```

---

## 🧠 Memory (Redis)

* Each chat session stores history in Redis under a unique `session_id`.
* Expiry (TTL) is configurable (e.g. **5 minutes**).
* If the user reloads the WordPress site → new session starts fresh.

---

---

## 📊 Architecture (Flow)

```mermaid
sequenceDiagram
    participant User as Website User
    participant WP as WordPress Plugin
    participant API as FastAPI RAG Backend
    participant Pinecone as Pinecone DB
    participant Redis as Redis Memory
    participant OpenAI as OpenAI GPT

    User->>WP: Ask a question
    WP->>API: POST /ask (company_id, session_id, question)
    API->>Redis: Load session history
    API->>Pinecone: Retrieve vectors (namespace=company)
    API->>OpenAI: Send context + history
    OpenAI->>API: Generate grounded answer
    API->>Redis: Update session memory
    API->>WP: Return answer
    WP->>User: Display chatbot response
```

---

## 📜 License

MIT License © 2025

---
