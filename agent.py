# ============================================
# Document AI Agent - Terminal Version v2.0
# Refactored to use core.py
# ============================================

import os
import time
import threading
from core import (
    load_config, create_client, load_documents_from_folder,
    build_retriever, load_persisted_retriever, ask,
    save_history, load_history, export_history, logger,
)

# ---- COLORS ----
GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

MEMORY_FILE = "chat_history.json"


def spinner(message, stop_event):
    """Show animated loading spinner."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        print(f"\r{CYAN}{frames[i]} {message}{RESET}", end="", flush=True)
        time.sleep(0.1)
        i = (i + 1) % len(frames)
    print("\r" + " " * (len(message) + 4) + "\r", end="")


def with_spinner(message, func, *args, **kwargs):
    """Run a function with a loading spinner."""
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=(message, stop))
    t.start()
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        stop.set()
        t.join()


def main():
    config = load_config()

    agent_line = f"{config['agent_name']} from {config['company_name']}"[:35]
    print(f"""
{GREEN}
╔══════════════════════════════════════╗
║     Document AI Agent v2.0  🤖      ║
║   {agent_line:<35s}║
╚══════════════════════════════════════╝
{RESET}""")

    # Initialize Groq client
    client = create_client()
    if not client:
        print(f"{RED}❌ No API key found!{RESET}")
        print(f"{YELLOW}💡 Create a .env file with: GROQ_API_KEY=your_key{RESET}")
        return

    # Try to load persisted retriever first
    print(f"\n{CYAN}🔍 Checking for previously loaded documents...{RESET}")
    retriever = with_spinner("Loading saved documents...", load_persisted_retriever)

    if retriever:
        print(f"{GREEN}✅ Loaded previously saved documents!{RESET}")
        print(f"{YELLOW}💡 Type 'reload' to load new documents{RESET}")
    else:
        print(f"{YELLOW}No saved documents found.{RESET}")
        print(f"{YELLOW}Enter the path to your documents folder:{RESET}")
        folder_path = input(">>> ").strip('"')

        documents = load_documents_from_folder(folder_path)
        if not documents:
            return

        for doc in documents:
            name = os.path.basename(doc.metadata.get("source", "unknown"))
            print(f"{GREEN}✅ Loaded: {name}{RESET}")

        retriever = with_spinner("Building AI memory...", build_retriever, documents)
        if not retriever:
            print(f"{RED}❌ Failed to build AI memory{RESET}")
            return

    # Load history
    history = load_history(MEMORY_FILE)
    if history:
        print(f"\n{CYAN}📝 Loaded {len(history)} previous conversations{RESET}")

    ready_line = f"{config['agent_name']} Ready! 🤖"[:35]
    print(f"""
{GREEN}
╔══════════════════════════════════════╗
║   {ready_line:<35s}║
║  Commands: exit, clear, export,      ║
║            reload                     ║
╚══════════════════════════════════════╝
{RESET}""")

    print(f"{GREEN}{config['agent_name']}:{RESET} {config['welcome_message']}\n")

    while True:
        print(f"{BLUE}You:{RESET} ", end="")
        try:
            question = input()
        except (EOFError, KeyboardInterrupt):
            save_history(history, MEMORY_FILE)
            print(f"\n{GREEN}Goodbye! 👋{RESET}")
            break

        cmd = question.lower().strip()

        if cmd == "exit":
            save_history(history, MEMORY_FILE)
            print(f"\n{GREEN}Goodbye! 👋{RESET}")
            break

        if cmd == "clear":
            history = []
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            print(f"{GREEN}✅ History cleared!{RESET}")
            continue

        if cmd == "export":
            result = export_history(history)
            print(f"{GREEN}{result}{RESET}")
            continue

        if cmd == "reload":
            print(f"{YELLOW}Enter the path to your documents folder:{RESET}")
            folder_path = input(">>> ").strip('"')
            documents = load_documents_from_folder(folder_path)
            if documents:
                for doc in documents:
                    name = os.path.basename(doc.metadata.get("source", "unknown"))
                    print(f"{GREEN}✅ Loaded: {name}{RESET}")
                new_retriever = with_spinner("Building AI memory...", build_retriever, documents)
                if new_retriever:
                    retriever = new_retriever
                    print(f"{GREEN}✅ Documents reloaded!{RESET}")
                else:
                    print(f"{RED}❌ Failed to rebuild. Keeping old documents.{RESET}")
            continue

        if not question.strip():
            continue

        answer, _ = with_spinner("Thinking...", ask, client, retriever, question, history, config)
        print(f"\n{GREEN}{config['agent_name']}:{RESET} {answer}\n")
        history.append((question, answer))
        save_history(history, MEMORY_FILE)


if __name__ == "__main__":
    main()