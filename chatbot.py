# chatbot.py

import streamlit as st
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "tinyllama",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # ✅ Initialize Embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs=self.encode_kwargs,
            )
        except Exception as e:
            st.error("❌ Failed to load HuggingFace embeddings.")
            raise e

        # ✅ Initialize LLM
        try:
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=self.llm_temperature,
            )
        except Exception as e:
            st.error("❌ Failed to connect to Ollama. Make sure Ollama is running and the model is pulled.")
            raise e

        # ✅ Prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # ✅ Qdrant Client
        try:
            self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)
            self.client.get_collection(self.collection_name)  # Ensure collection exists
        except Exception as e:
            st.error("❌ Failed to connect to Qdrant. Please ensure Qdrant is running (localhost:6333).")
            raise e

        # ✅ Vector Store
        try:
            self.db = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=self.collection_name,
            )
        except Exception as e:
            st.error("❌ Failed to initialize vector store from Qdrant.")
            raise e

        # ✅ Retriever + QA Chain
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt},
            verbose=False
        )

    def get_response(self, query: str) -> str:
        try:
            result = self.qa.invoke({"query": query})
            return result if isinstance(result, str) else result.get("result", "")
        except Exception as e:
            st.error(f"❌ Error during response generation: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."
