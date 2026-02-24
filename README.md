# 📄 PDF Chat & Summarization App  
### Streamlit + LangChain + Groq (Llama 3.1) + FAISS

---

## 🚀 Overview

This project is an interactive **PDF Summarization and Conversational AI Application** built using:

- Streamlit (Frontend UI)
- LangChain (LLM Orchestration)
- Groq – Llama 3.1 8B Instant
- HuggingFace Embeddings
- FAISS Vector Store

The application allows users to:

- 📄 Upload a PDF document  
- 🧠 Generate an AI-powered summary  
- 💬 Ask contextual questions about the document  
- 🗂 Maintain session-based chat history  

---

## 🏗 Architecture Flow

1. User uploads a PDF  
2. PDF is loaded using PyPDFLoader  
3. Text is split using RecursiveCharacterTextSplitter  
4. Summary generated using Map-Reduce Summarization Chain  
5. Chunks embedded using sentence-transformers/all-MiniLM-L6-v2  
6. Stored in FAISS vector database  
7. Chat handled using RunnableWithMessageHistory  

---

## 🧠 Key Features

- PDF Upload & Summarization  
- Map-Reduce LLM Summarization  
- Adjustable Temperature & Max Tokens  
- FAISS Vector Indexing  
- Session-based Chat Memory  
- Groq Llama 3.1 Integration  

---

## 🧰 Tech Stack

- Python  
- Streamlit  
- LangChain  
- Groq API  
- HuggingFace Embeddings  
- FAISS    

---

## ⚙ LLM Configuration

Model Used:

```
llama-3.1-8b-instant
```

Adjustable Parameters:
- Temperature (0.0 – 2.0)
- Max Tokens (10 – 500)

---


### 2️⃣ Create Virtual Environment (Recommended)

Windows:
```bash
conda init
conda activate .\venv
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

---

## ▶ Run the Application

```bash
streamlit run app.py
```

App runs on:

```
http://localhost:8501
```

---


---

## 🔍 How It Works

### Summarization
- Uses LangChain load_summarize_chain
- Chain type: map_reduce
- Efficient for large PDFs

### Embeddings
- Model: all-MiniLM-L6-v2
- Converts text chunks into vector embeddings

### Vector Store
- FAISS used for semantic similarity search
- Top-k retrieval = 3

### Chat Memory
- Session-based history
- Managed using RunnableWithMessageHistory

---

## 🎯 Use Cases

- Research paper summarization  
- Resume review  
- Legal document analysis  
- Academic preparation  
- Business report understanding  

---

## 🔮 Future Enhancements

- Full Retrieval-Augmented Generation (RAG) pipeline  
- Multi-PDF support  
- Persistent database-backed memory  
- AWS / Azure deployment  
- Docker containerization  

---

## 👨‍💻 Author

**Nihal Radhakrishna**  
Data Engineering & AI Enthusiast  
