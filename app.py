# ============================================
# Document AI Agent - Web Version v3.0
# Refactored to use core.py
# ============================================

import os
import json
import gradio as gr
from core import (
    load_config, create_client, load_documents_from_files,
    build_retriever, load_persisted_retriever, ask,
    save_history, export_history, logger,
)

# ---- GLOBALS ----
retriever = None
client = None
MEMORY_FILE = "web_chat_history.json"


def load_documents(files):
    """Load uploaded documents and build retriever."""
    global retriever

    if not files:
        return "⚠️ Please upload at least one document!"

    file_paths = [f.name for f in files]
    documents, loaded, failed = load_documents_from_files(file_paths)

    if not documents:
        return "❌ No supported documents loaded!\n💡 Supported: PDF, DOCX, TXT"

    new_retriever = build_retriever(documents)
    if not new_retriever:
        return "❌ Failed to build AI memory"

    retriever = new_retriever

    status = f"✅ Loaded {len(loaded)} document(s):\n"
    status += "\n".join([f"📄 {name}" for name in loaded])
    if failed:
        status += f"\n\n⚠️ Failed:\n" + "\n".join([f"❌ {f}" for f in failed])
    return status


def chat(message, history):
    """Process question and return answer.
    
    Gradio ChatInterface passes `history` as a list of
    {"role": ..., "content": ...} dicts (Gradio >=4 messages format).
    We forward that directly to core.ask().
    """
    global retriever, client

    config = load_config()

    if client is None:
        client_obj = create_client()
        if client_obj is None:
            return "❌ No API key found! Create a .env file with: GROQ_API_KEY=your_key"
        client = client_obj

    answer, _ = ask(client, retriever, message, history, config)

    # Save history
    save_history(
        history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}],
        MEMORY_FILE,
    )
    return answer


def do_export():
    """Export chat history."""
    history = []
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
    except Exception:
        pass
    return export_history(history)


# ---- Initialize ----
config = load_config()
client = create_client()

# Try to load persisted retriever
retriever = load_persisted_retriever()
persisted_status = ""
if retriever:
    persisted_status = "✅ Previously uploaded documents are still loaded! You can start chatting."
    logger.info("Loaded persisted retriever on startup")

# ---- BUILD INTERFACE ----
# Use Blocks with a manual Chatbot + Textbox instead of nesting
# ChatInterface inside Blocks (which causes layout/render errors).
with gr.Blocks(
    title=f"{config['agent_name']} - Document AI Agent v2.0",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(f"""
    # 🤖 {config['agent_name']} — {config['company_name']}
    ### Powered by Document AI Agent v2.0
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Documents")
            file_upload = gr.File(
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"],
                label="Drop your documents here",
            )
            upload_btn = gr.Button(
                "🚀 Load Documents",
                variant="primary",
            )
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=5,
                value=persisted_status,
            )
            upload_btn.click(
                fn=load_documents,
                inputs=[file_upload],
                outputs=[upload_status],
            )

            gr.Markdown("### 💾 Export Chat")
            export_btn = gr.Button(
                "📥 Export History",
                variant="secondary",
            )
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False,
                lines=3,
            )
            export_btn.click(
                fn=do_export,
                outputs=[export_status],
            )

            gr.Markdown("### 💡 Tips")
            gr.Markdown(f"""
            - Upload PDF, DOCX or TXT files
            - Ask in ANY language
            - {config['agent_name']} replies in your language
            - Documents persist between restarts
            - Export saves TXT and JSON
            """)

        with gr.Column(scale=2):
            gr.Markdown(f"### 💬 Chat with {config['agent_name']}")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                type="messages",
            )
            msg_input = gr.Textbox(
                placeholder="Type your message here...",
                label="Your message",
                lines=1,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            # Example questions
            gr.Examples(
                examples=[
                    "What are your opening hours?",
                    "What is your return policy?",
                    "Quelles sont vos heures d'ouverture?",
                    "¿Cuáles son sus horarios?",
                    "How do I contact support?",
                ],
                inputs=msg_input,
            )

    def user_send(message, history):
        """Handle user message: get answer, update chatbot."""
        if not message.strip():
            return "", history
        answer = chat(message, history)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return "", history

    def clear_chat():
        return [], ""

    send_btn.click(
        fn=user_send,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )
    msg_input.submit(
        fn=user_send,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_input],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
