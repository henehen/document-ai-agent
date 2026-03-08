# ============================================
# Document AI Agent - Terminal Version v1.2
# Author: Henrique Faria Cl
# Changes: Multiple languages, custom
#          personality, chat export
# ============================================

import os
import json
import time
import threading
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

# ---- COLORS ----
GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

# ---- FILES ----
MEMORY_FILE = "chat_history.json"
CONFIG_FILE = "config.json"

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
                return default
        else:
            print(f"{YELLOW}⚠️ No config.json found — using defaults{RESET}")
            return default
    except Exception as e:
        print(f"{RED}❌ Error loading config: {str(e)}{RESET}")
        return default

def save_history(history):
    """Save conversation history to file"""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"{RED}❌ Failed to save history: {str(e)}{RESET}")

def load_history():
    """Load conversation history from file"""
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def export_history(history):
    """Export chat history to TXT and JSON"""
    if not history:
        print(f"{YELLOW}⚠️ No conversation history to export{RESET}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export as TXT
    txt_file = f"export_{timestamp}.txt"
    try:
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"Chat History Export\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            for i, (question, answer) in enumerate(history, 1):
                f.write(f"[{i}] Customer: {question}\n")
                f.write(f"    Agent: {answer}\n\n")
        print(f"{GREEN}✅ Exported as text: {txt_file}{RESET}")
    except Exception as e:
        print(f"{RED}❌ Failed to export TXT: {str(e)}{RESET}")

    # Export as JSON
    json_file = f"export_{timestamp}.json"
    try:
        export_data = {
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_messages": len(history),
            "conversations": [
                {"question": q, "answer": a}
                for q, a in history
            ]
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"{GREEN}✅ Exported as JSON: {json_file}{RESET}")
    except Exception as e:
        print(f"{RED}❌ Failed to export JSON: {str(e)}{RESET}")

def spinner(message, stop_event):
    """Show animated loading spinner"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        print(f"\r{CYAN}{frames[i]} {message}{RESET}", end="", flush=True)
        time.sleep(0.1)
        i = (i + 1) % len(frames)
    print("\r" + " " * (len(message) + 4) + "\r", end="")

def load_documents(folder_path):
    """Load all documents from folder"""
    if not os.path.exists(folder_path):
        print(f"{RED}❌ Folder not found: {folder_path}{RESET}")
        print(f"{YELLOW}💡 Please check the path and try again{RESET}")
        return None

    documents = []
    supported = {
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt":  TextLoader
    }

    print(f"\n{BLUE}📂 Scanning folder: {folder_path}{RESET}")

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported:
            filepath = os.path.join(folder_path, filename)
            try:
                loader = supported[ext](filepath)
                documents.extend(loader.load())
                print(f"{GREEN}✅ Loaded: {filename}{RESET}")
            except Exception as e:
                print(f"{RED}❌ Failed to load {filename}: {str(e)}{RESET}")

    if not documents:
        print(f"{RED}❌ No supported documents found!{RESET}")
        print(f"{YELLOW}💡 Supported formats: PDF, DOCX, TXT{RESET}")
        return None

    print(f"{GREEN}✅ Documents loaded successfully!{RESET}")
    return documents

def build_memory(documents):
    """Build AI searchable memory"""
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=("Building AI memory...", stop))
    t.start()
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        stop.set()
        t.join()
        print(f"{GREEN}✅ Memory ready! ({len(chunks)} chunks){RESET}")
        return vectorstore.as_retriever()
    except Exception as e:
        stop.set()
        t.join()
        print(f"{RED}❌ Failed to build memory: {str(e)}{RESET}")
        return None

def ask_agent(client, retriever, question, history, config):
    """Get answer from AI with custom personality and language detection"""
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=("Thinking...", stop))
    t.start()

    try:
        relevant_docs = retriever.invoke(question)
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
- Never switch languages unless the customer does

Answer questions based ONLY on this business information:
{context}

Rules:
- Always be {config['tone']} and helpful
- Keep answers clear and concise
- If the answer isn't in the documents, say: "{config['unknown_answer']}"
- Never make up information
- Sign off as {config['agent_name']} from {config['company_name']}"""
            }
        ]

        for human, assistant in history[-6:]:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})

        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500
        )

        stop.set()
        t.join()
        return response.choices[0].message.content

    except Exception as e:
        stop.set()
        t.join()
        if "401" in str(e):
            return f"{RED}❌ Invalid API key. Please check your .env file{RESET}"
        elif "429" in str(e):
            return f"{RED}❌ Rate limit reached. Please wait a moment{RESET}"
        elif "503" in str(e):
            return f"{RED}❌ AI service temporarily unavailable. Try again{RESET}"
        else:
            return f"{RED}❌ Error: {str(e)}{RESET}"

def main():
    # Load config
    config = load_config()

    print(f"""
{GREEN}
╔══════════════════════════════════════╗
║     Document AI Agent v1.2  🤖       ║
║   {config['agent_name']} from {config['company_name'][:20]}
╚══════════════════════════════════════╝
{RESET}""")

    if not GROQ_API_KEY:
        print(f"{RED}❌ No API key found!{RESET}")
        print(f"{YELLOW}💡 Create a .env file with: GROQ_API_KEY=your_key{RESET}")
        return

    print(f"{YELLOW}Enter the path to your documents folder:{RESET}")
    folder_path = input(">>> ").strip('"')

    documents = load_documents(folder_path)
    if not documents:
        return

    retriever = build_memory(documents)
    if not retriever:
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"{RED}❌ Failed to connect to Groq: {str(e)}{RESET}")
        return

    history = load_history()
    if history:
        print(f"\n{CYAN}📝 Loaded {len(history)} previous conversations{RESET}")

    print(f"""
{GREEN}
╔══════════════════════════════════════╗
║         {config['agent_name']} Ready! 🤖              
║  'exit' quit 'clear' reset 'export'  ║
╚══════════════════════════════════════╝
{RESET}""")

    print(f"{GREEN}{config['agent_name']}:{RESET} {config['welcome_message']}\n")

    while True:
        print(f"{BLUE}You:{RESET} ", end="")
        question = input()

        if question.lower() == "exit":
            save_history(history)
            print(f"\n{GREEN}Goodbye! 👋{RESET}")
            break

        if question.lower() == "clear":
            history = []
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            print(f"{GREEN}✅ History cleared!{RESET}")
            continue

        if question.lower() == "export":
            export_history(history)
            continue

        if not question.strip():
            continue

        answer = ask_agent(client, retriever, question, history, config)
        print(f"\n{GREEN}{config['agent_name']}:{RESET} {answer}\n")
        history.append((question, answer))
        save_history(history)

if __name__ == "__main__":
    main()