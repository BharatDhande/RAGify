# app.py

import streamlit as st
from streamlit import session_state
import time
import base64
import os
from vectors import EmbeddingsManager
from chatbot import ChatbotManager

# Display PDF
def displayPDF(file):
    base64_pdf = base64.b64encode(file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# Check if Qdrant is reachable
@st.cache_resource
def check_qdrant_connection():
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(url="http://localhost:6333")
        client.get_collections()
        return True
    except Exception:
        return False

# Initialize session state
st.session_state.setdefault('temp_pdf_path', None)
st.session_state.setdefault('chatbot_manager', None)
st.session_state.setdefault('messages', [])

# Layout
st.set_page_config(page_title="RAGify App", layout="wide")

# Sidebar
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.warning("⚠️ Logo image not found.")
    st.markdown("### 📚 Your Personal Document Assistant")
    st.markdown("---")
    menu = ["🏠 Home", "🤖 Chatbot", "📧 Contact"]
    choice = st.radio("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("📄 RAGify App")
    st.markdown("""
Welcome to **RAGify**! 🚀

**Tech stack**: Llama 3.2 + BGE Embeddings + Qdrant (via Docker)

- 📤 Upload your PDFs  
- 🧠 Create Embeddings  
- 🤖 Chat with the contents

Boost your document understanding now!
""")

# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Chat with Your PDF")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # 📂 Upload PDF
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            st.success("📄 File uploaded!")
            st.markdown(f"**Filename:** `{uploaded_file.name}`")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.markdown("### PDF Preview")
            uploaded_file.seek(0)  # ✅ FIXED: Reset pointer before reading again
            displayPDF(uploaded_file)

            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['temp_pdf_path'] = temp_path

    # 🧠 Embedding Section
    with col2:
        st.header("🧠 Create Embeddings")
        create_embeddings = st.button("✅ Create Embeddings")
        if create_embeddings:
            if not st.session_state['temp_pdf_path']:
                st.warning("⚠️ Upload a PDF first.")
            elif not check_qdrant_connection():
                st.error("❌ Qdrant server not reachable on `localhost:6333`. Please ensure Docker is running:")
                st.code("docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant", language="bash")
            else:
                try:
                    with st.spinner("🔄 Generating Embeddings..."):
                        embeddings_manager = EmbeddingsManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            qdrant_url="http://localhost:6333",
                            collection_name="vector_db"
                        )
                        msg = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                        time.sleep(1)
                    st.success(msg)

                    if not st.session_state['chatbot_manager']:
                        st.session_state['chatbot_manager'] = ChatbotManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_model="tinyllama",
                            llm_temperature=0.7,
                            qdrant_url="http://localhost:6333",
                            collection_name="vector_db"
                        )
                except Exception as e:
                    st.error("❌ Failed to generate embeddings.")
                    st.exception(e)

    # 💬 Chat Interface
    with col3:
        st.header("💬 Ask Questions")
        if not st.session_state['chatbot_manager']:
            st.info("ℹ️ Please upload a PDF and generate embeddings first.")
        else:
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            if user_input := st.chat_input("Ask your question..."):
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                try:
                    with st.spinner("🤖 Thinking..."):
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)
                except Exception as e:
                    answer = f"⚠️ An error occurred: `{e}`"
                st.chat_message("assistant").markdown(answer)
                st.session_state['messages'].append({"role": "assistant", "content": answer})


# Footer
st.markdown("---")
st.markdown("© 2024 RAGify. All rights reserved.")
