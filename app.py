# ============================================
# Document AI Agent - Web Version v1.1
# Author: Henrique Faria Cl
# Changes: Persistent memory, better errors,
#          loading spinner
# ============================================

import os
import json
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---- LOAD API KEYS ----
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- GLOBALS ----
retriever = None
client = None
MEMORY_FILE = "web_chat_history.json"

def save_history(history):
    """Save conversation history to file"""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

def load_history():
    """Load conversation history from file"""
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def initialize_client():
    """Initialize Groq client with error handling"""
    global client
    if not GROQ_API_KEY:
        return "❌ No API key found! Create a .env file with: GROQ_API_KEY=your_key"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return None
    except Exception as e:
        return f"❌ Failed to connect to Groq: {str(e)}"

def load_documents(files):
    """Load uploaded documents"""
    global retriever

    if not files:
        return "⚠️ Please upload at least one document!"

    documents = []
    supported = {
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt":  TextLoader
    }

    loaded = []
    failed = []

    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in supported:
            try:
                loader = supported[ext](file.name)
                documents.extend(loader.load())
                loaded.append(os.path.basename(file.name))
            except Exception as e:
                failed.append(f"{os.path.basename(file.name)}: {str(e)}")
        else:
            failed.append(f"{os.path.basename(file.name)}: unsupported format")

    if not documents:
        return "❌ No supported documents loaded!\n💡 Supported formats: PDF, DOCX, TXT"

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        return f"❌ Failed to build AI memory: {str(e)}"

    status = f"✅ Successfully loaded {len(loaded)} document(s):\n"
    status += "\n".join([f"📄 {name}" for name in loaded])

    if failed:
        status += f"\n\n⚠️ Failed to load:\n"
        status += "\n".join([f"❌ {f}" for f in failed])

    return status

def chat(message, history):
    """Process question and return answer"""
    global retriever, client

    if not message.strip():
        return "Please type a question!"

    if retriever is None:
        return "⚠️ Please upload your documents first!"

    if client is None:
        error = initialize_client()
        if error:
            return error

    try:
        relevant_docs = retriever.invoke(message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            {
                "role": "system",
                "content": f"""You are a professional and friendly customer service agent.
Answer questions based ONLY on the documents provided.

Documents content:
{context}

Rules:
- Always be friendly and professional
- Keep answers clear and concise
- If the answer isn't in the documents, politely say so
- Suggest contacting support for unknown questions
- Never make up information"""
            }
        ]

        for msg in history[-6:]:
            if isinstance(msg, dict):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            else:
                human, assistant = msg
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        save_history(history)
        return answer

    except Exception as e:
        if "401" in str(e):
            return "❌ Invalid API key. Please check your .env file."
        elif "429" in str(e):
            return "⏳ Rate limit reached. Please wait a moment and try again."
        elif "503" in str(e):
            return "🔄 AI service temporarily unavailable. Please try again."
        else:
            return f"❌ Something went wrong: {str(e)}"

# ---- Initialize client on startup ----
initialize_client()

# ---- Load previous history ----
previous_history = load_history()

# ---- BUILD INTERFACE ----
with gr.Blocks(title="Document AI Agent v1.1") as demo:

    gr.Markdown("""
    # 🤖 Document AI Agent v1.1
    ### Upload your business documents and let AI answer customer questions instantly!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Documents")
            file_upload = gr.File(
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"],
                label="Drop your documents here"
            )
            upload_btn = gr.Button(
                "🚀 Load Documents",
                variant="primary"
            )
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=5
            )
            upload_btn.click(
                fn=load_documents,
                inputs=[file_upload],
                outputs=[upload_status]
            )

            gr.Markdown("### 💡 Tips")
            gr.Markdown("""
            - Upload PDF, DOCX or TXT files
            - Ask questions in plain English
            - Agent answers based on YOUR documents
            - Unknown questions redirect to support
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat With Your Documents")
            chatbot = gr.ChatInterface(
                fn=chat,
                examples=[
                    "What are your opening hours?",
                    "What is your return policy?",
                    "How do I contact support?",
                    "What payment methods do you accept?",
                    "How long does delivery take?"
                ],
            )

demo.launch(share=True)