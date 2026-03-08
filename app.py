# ============================================
# Document AI Agent - Web Interface Version
# ============================================

import os
import gradio as gr
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---- YOUR API KEY ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ---- GLOBAL VARIABLES ----
retriever = None
client = Groq(api_key=GROQ_API_KEY)

def load_documents(files):
    """Load uploaded documents and build AI memory"""
    global retriever

    if not files:
        return "❌ Please upload at least one document!"

    documents = []
    supported = {
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt":  TextLoader
    }

    loaded = []
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in supported:
            loader = supported[ext](file.name)
            documents.extend(loader.load())
            loaded.append(os.path.basename(file.name))

    if not documents:
        return "❌ No supported documents found! Use PDF, DOCX or TXT files."

    # Build AI memory
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    return f"✅ Successfully loaded {len(loaded)} document(s):\n" + "\n".join([f"📄 {name}" for name in loaded])

def chat(message, history):
    """Process customer question and return answer"""
    global retriever

    if retriever is None:
        return "⚠️ Please upload your documents first using the panel on the left!"

    if not message.strip():
        return "Please type a question!"

    # Get relevant document chunks
    relevant_docs = retriever.invoke(message)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build messages
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

    # Add chat history
    for human, assistant in history[-3:]:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    # Get answer
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=500
    )

    return response.choices[0].message.content

# ---- BUILD WEB INTERFACE ----
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Document AI Agent"
) as demo:

    gr.Markdown("""
    # 🤖 Document AI Agent
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
                lines=4
            )
            upload_btn.click(
                fn=load_documents,
                inputs=[file_upload],
                outputs=[upload_status]
            )

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