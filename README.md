# 🧠 RAGify – Website RAG Chatbot

RAGify is a Retrieval-Augmented Generation (RAG) based chatbot powered by **LangChain**, **Google's Generative AI**, and a **Gradio** UI. It intelligently answers user queries using content from pre-processed websites, providing contextually relevant and fact-based responses.

---

## 🚀 Features

- ✅ Retrieval-Augmented Generation (RAG) architecture
- ⚙️ FAISS vector database for fast semantic search
- 🧩 LangChain for modular pipeline management
- 🤖 Google Generative AI for natural language generation
- 🎨 Gradio interface with custom styling

---

## 🧱 Tech Stack

- **LangChain**
- **Google Generative AI**
- **Gradio**
- **FAISS**
- **Python 3.9+**
- **Docker + Google Cloud Run**

---

## 📁 Project Structure
── app.py # Main app logic for chatbot + UI
├── requirements.txt # Python dependencies
├── Dockerfile # Docker config for deployment
├── .env # Environment variables (not included)
├── Vector_DB/ # FAISS index storage
├── Data/ # Raw or cleaned data
├── Flask/ # (Optional) Flask components, if used
├── Dialogram.excalidraw.svg # Architecture diagram
└── data_raw.py # Data preprocessing script


---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/BharatDhande/RAGify.git
cd RAGify

pip install -r requirements.txt
GOOGLE_API_KEY=your_google_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
python app.py


