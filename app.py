# ============================================
# Document AI Agent - Web Version v1.2
# Author: Henrique Faria Cl
# Changes: Multiple languages, custom
#          personality, chat export
# ============================================

import os
import json
import gradio as gr
from datetime import datetime
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
CONFIG_FILE = "config.json"
MEMORY_FILE = "web_chat_history.json"

def load_config():
    """Load business configuration"""
    default = {
        "agent_name": "Agent",
        "company_name": "Our Business",
        "tone": "professional",
        "support_email": "support@business.com",
        "welcome_message": "Hello! How can I help you today?",
        "unknown_answer": "I'm sorry, I don't have that information. Please contact our support team."
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                default.update(config)
    except Exception:
        pass
    return default

def save_history(history):
    """Save conversation history"""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

def export_history(history):
    """Export chat history to TXT and JSON files"""
    if not history:
        return "⚠️ No conversation history to export!"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exported = []

    # Export TXT
    txt_file = f"export_{timestamp}.txt"
    try:
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"Chat History Export\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            for i, msg in enumerate(history, 1):
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    f.write(f"[{i}] {role.capitalize()}: {content}\n\n")
                else:
                    q, a = msg
                    f.write(f"Customer: {q}\nAgent: {a}\n\n")
        exported.append(f"📄 {txt_file}")
    except Exception as e:
        exported.append(f"❌ TXT failed: {str(e)}")

    # Export JSON
    json_file = f"export_{timestamp}.json"
    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "conversations": history
            }, f, indent=2, ensure_ascii=False)
        exported.append(f"📊 {json_file}")
    except Exception as e:
        exported.append(f"❌ JSON failed: {str(e)}")

    return "✅ Exported successfully!\n" + "\n".join(exported)

def initialize_client():
    """Initialize Groq client"""
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
        return "❌ No supported documents loaded!\n💡 Supported: PDF, DOCX, TXT"

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        return f"❌ Failed to build AI memory: {str(e)}"

    status = f"✅ Loaded {len(loaded)} document(s):\n"
    status += "\n".join([f"📄 {name}" for name in loaded])
    if failed:
        status += f"\n\n⚠️ Failed:\n" + "\n".join([f"❌ {f}" for f in failed])
    return status

def chat(message, history):
    """Process question and return answer"""
    global retriever, client

    config = load_config()

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
                "content": f"""You are {config['agent_name']}, a customer service agent for {config['company_name']}.
Your tone is {config['tone']}.

IMPORTANT LANGUAGE RULE:
- Detect the language the customer is writing in
- Always respond in the SAME language as the customer
- If they write in French, respond in French
- If they write in Spanish, respond in Spanish
- If they write in English, respond in English

Answer questions based ONLY on this business information:
{context}

Rules:
- Always be {config['tone']} and helpful
- Keep answers clear and concise
- If answer isn't in documents say: "{config['unknown_answer']}"
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
        save_history(history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        return answer

    except Exception as e:
        if "401" in str(e):
            return "❌ Invalid API key. Please check your .env file."
        elif "429" in str(e):
            return "⏳ Rate limit reached. Please wait a moment."
        elif "503" in str(e):
            return "🔄 AI service temporarily unavailable."
        else:
            return f"❌ Something went wrong: {str(e)}"

# ---- Initialize ----
config = load_config()
initialize_client()

# ---- BUILD INTERFACE ----
with gr.Blocks(title=f"{config['agent_name']} - Document AI Agent v1.2") as demo:

    gr.Markdown(f"""
    # 🤖 {config['agent_name']} — {config['company_name']}
    ### Powered by Document AI Agent v1.2
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

            gr.Markdown("### 💾 Export Chat")
            export_btn = gr.Button(
                "📥 Export History",
                variant="secondary"
            )
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False,
                lines=3
            )

            gr.Markdown("### 💡 Tips")
            gr.Markdown(f"""
            - Upload PDF, DOCX or TXT files
            - Ask in ANY language
            - {config['agent_name']} replies in your language
            - Export saves TXT and JSON
            """)

        with gr.Column(scale=2):
            gr.Markdown(f"### 💬 Chat with {config['agent_name']}")
            chatbot = gr.ChatInterface(
                fn=chat,
                examples=[
                    "What are your opening hours?",
                    "What is your return policy?",
                    "Quelles sont vos heures d'ouverture?",
                    "¿Cuáles son sus horarios?",
                    "How do I contact support?"
                ],
            )
            export_btn.click(
    fn=lambda: export_history(
        json.load(open(MEMORY_FILE)) 
        if os.path.exists(MEMORY_FILE) 
        else []
    ),
    outputs=[export_status]
)

demo.launch(share=True)