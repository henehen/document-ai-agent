# ============================================
# Document AI Agent - Terminal Version
# Author: Henrique Faria Cl
# Description: AI agent that reads documents
#              and answers questions about them
# ============================================

import os
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---- LOAD API KEYS ----
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")# ---- COLORS FOR TERMINAL ----
GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def load_documents(folder_path):
    """
    Loads all PDF, Word and TXT documents
    from a folder automatically
    """
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
            print(f"{GREEN}✅ Loading: {filename}{RESET}")
            loader = supported[ext](filepath)
            documents.extend(loader.load())

    if not documents:
        print(f"{RED}❌ No documents found in folder!{RESET}")
        print(f"Supported formats: PDF, DOCX, TXT")
        exit()

    print(f"{GREEN}✅ Loaded {len(documents)} document(s) successfully!{RESET}")
    return documents

def build_memory(documents):
    """
    Splits documents into chunks and
    builds AI searchable memory
    """
    print(f"\n{BLUE}🧠 Building AI memory...{RESET}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"{GREEN}✅ Split into {len(chunks)} chunks{RESET}")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print(f"{GREEN}✅ Memory ready!{RESET}")

    return vectorstore.as_retriever()

def ask_agent(client, retriever, question, history):
    """
    Sends question to AI with document
    context and returns answer
    """
    # Get relevant document chunks
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build message history
    messages = [
        {
            "role": "system",
            "content": f"""You are a professional and friendly customer service agent.
Your job is to answer questions based ONLY on the documents provided.

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

    # Add last 3 exchanges for memory
    for human, assistant in history[-3:]:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    # Add current question
    messages.append({"role": "user", "content": question})

    # Get answer from Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=500
    )

    return response.choices[0].message.content

def main():
    print(f"""
{GREEN}
╔══════════════════════════════════════╗
║       Document AI Agent v1.0         ║
║   Ask anything about your documents  ║
╚══════════════════════════════════════╝
{RESET}""")

    # Get documents folder
    print(f"{YELLOW}Enter the path to your documents folder:{RESET}")
    folder_path = input(">>> ").strip('"')

    # Load and process documents
    documents = load_documents(folder_path)
    retriever = build_memory(documents)

    # Connect to Groq
    client = Groq(api_key=GROQ_API_KEY)

    # Start chat
    print(f"""
{GREEN}
╔══════════════════════════════════════╗
║         Agent Ready! 🤖              ║
║   Type 'exit' to quit                ║
╚══════════════════════════════════════╝
{RESET}""")

    history = []

    while True:
        print(f"{BLUE}You:{RESET} ", end="")
        question = input()

        if question.lower() == "exit":
            print(f"\n{GREEN}Goodbye! Have a great day! 👋{RESET}")
            break

        if not question.strip():
            continue

        answer = ask_agent(client, retriever, question, history)
        print(f"\n{GREEN}Agent:{RESET} {answer}\n")

        history.append((question, answer))

if __name__ == "__main__":
    main()