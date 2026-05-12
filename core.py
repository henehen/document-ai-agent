# ============================================
# Document AI Agent - Core Module v2.0
# Shared logic for all interfaces
# (CLI, Gradio, FastAPI)
# ============================================

import os
import json
import logging
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---- LOGGING ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("document-ai-agent")

# ---- LOAD ENV ----
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# ---- CONSTANTS ----
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

# Use a multilingual model so French/Spanish/Portuguese queries
# match English documents (and vice versa)
# Sentinel the LLM is asked to prepend when it cannot answer from the
# provided documents. We strip it before returning the answer to the user
# and use it as a language-agnostic signal for analytics.
UNANSWERED_MARKER = "[NO_ANSWER]"


# ======================
# Configuration
# ======================

def load_config() -> dict:
    """Load business configuration from config.json with sensible defaults."""
    defaults = {
        "agent_name": "Agent",
        "company_name": "Our Business",
        "tone": "professional",
        "support_email": "support@business.com",
        "welcome_message": "Hello! How can I help you today?",
        "unknown_answer": "I'm sorry, I don't have that information. Please contact our support team.",
        "allowed_origins": ["*"],
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                defaults.update(config)
                logger.info("Loaded config from %s", CONFIG_FILE)
        else:
            logger.warning("No config.json found — using defaults")
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config.json: %s — using defaults", e)
    except Exception as e:
        logger.error("Failed to load config: %s — using defaults", e)
    return defaults


# ======================
# Groq Client
# ======================

def create_client() -> Groq | None:
    """Create and return a Groq client, or None if the key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment")
        return None
    try:
        client = Groq(api_key=api_key)
        logger.info("Groq client initialized")
        return client
    except Exception as e:
        logger.error("Failed to create Groq client: %s", e)
        return None


# ======================
# Document Loading
# ======================

def load_documents_from_folder(folder_path: str) -> list:
    """Load all supported documents from a local folder. Returns list of Document objects."""
    if not os.path.exists(folder_path):
        logger.error("Folder not found: %s", folder_path)
        return []

    documents = []
    loaded = []
    failed = []

    logger.info("Scanning folder: %s", folder_path)
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            filepath = os.path.join(folder_path, filename)
            try:
                loader = SUPPORTED_EXTENSIONS[ext](filepath)
                documents.extend(loader.load())
                loaded.append(filename)
                logger.info("Loaded: %s", filename)
            except Exception as e:
                failed.append(filename)
                logger.warning("Failed to load %s: %s", filename, e)

    if not documents:
        logger.warning("No supported documents found in %s", folder_path)
    else:
        logger.info("Loaded %d document(s), %d failed", len(loaded), len(failed))
    return documents


def load_documents_from_files(file_paths: list[str]) -> tuple[list, list[str], list[str]]:
    """
    Load documents from a list of file paths (for web upload).
    Returns (documents, loaded_names, failed_names).
    """
    documents = []
    loaded = []
    failed = []

    for filepath in file_paths:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            try:
                loader = SUPPORTED_EXTENSIONS[ext](filepath)
                documents.extend(loader.load())
                loaded.append(os.path.basename(filepath))
            except Exception as e:
                failed.append(f"{os.path.basename(filepath)}: {e}")
                logger.warning("Failed to load %s: %s", filepath, e)
        else:
            failed.append(f"{os.path.basename(filepath)}: unsupported format")

    logger.info("Loaded %d file(s), %d failed", len(loaded), len(failed))
    return documents, loaded, failed


# ======================
# Retriever (TF-IDF)
# ======================

class TFIDFRetriever:
    """Lightweight in-memory retriever using TF-IDF + cosine similarity."""

    def __init__(self, documents: list, k: int = 4):
        self.documents = documents
        self.k = k
        self.vectorizer = TfidfVectorizer()
        texts = [doc.page_content for doc in documents]
        self.matrix = self.vectorizer.fit_transform(texts)

    def invoke(self, query: str) -> list:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_k = np.argsort(scores)[-self.k:][::-1]
        return [self.documents[i] for i in top_k]


def build_retriever(documents: list, persist: bool = True):
    """
    Build a TF-IDF retriever from documents.
    The persist parameter is kept for API compatibility but is unused
    (TF-IDF is in-memory only).
    Returns a TFIDFRetriever or None on failure.
    """
    if not documents:
        logger.error("No documents provided to build retriever")
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)

        # Filter out empty or whitespace-only chunks
        chunks = [c for c in chunks if c.page_content.strip()]
        logger.info("Split into %d non-empty chunks", len(chunks))

        if not chunks:
            logger.error("All chunks were empty — document may be unreadable or image-based")
            return None

        retriever = TFIDFRetriever(chunks)
        logger.info("TF-IDF retriever ready (%d chunks)", len(chunks))
        return retriever

    except Exception as e:
        logger.error("Failed to build retriever: %s", e, exc_info=True)
        return None


def load_persisted_retriever():
    """TF-IDF is in-memory only — nothing to load from disk."""
    return None


# ======================
# Chat / Ask
# ======================

def build_system_prompt(config: dict, context: str) -> str:
    """Build the system prompt with config and retrieved context."""
    return f"""You are {config['agent_name']}, a customer service agent for {config['company_name']}.
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
- If the answer is NOT in the documents, your reply MUST begin with the exact token {UNANSWERED_MARKER} (always in English, with the brackets, no quotes), followed by a space, followed by "{config['unknown_answer']}" translated into the customer's language. Do not include the token anywhere else.
- Never make up information
- When possible, mention which document or section the answer comes from"""


def ask(
    client: Groq,
    retriever,
    question: str,
    history: list,
    config: dict,
) -> tuple[str, bool]:
    """
    Get an answer from the AI agent.
    
    Args:
        client: Groq client
        retriever: vector store retriever
        question: user's question
        history: list of (question, answer) tuples or dicts
        config: loaded config dict
    
    Returns:
        (answer, answered) where `answered` is False when the agent could not
        answer from the provided documents (the LLM emitted UNANSWERED_MARKER).
    """
    if not question.strip():
        return "Please type a question!", True

    if retriever is None:
        return "⚠️ Please upload your documents first!", True

    if client is None:
        return "❌ AI client not initialized. Check your API key.", True

    try:
        # Retrieve relevant documents
        relevant_docs = retriever.invoke(question)
        
        # Build context with source info
        context_parts = []
        for doc in relevant_docs:
            source = doc.metadata.get("source", "unknown")
            source_name = os.path.basename(source)
            page = doc.metadata.get("page", "")
            page_info = f" (page {page + 1})" if page != "" else ""
            context_parts.append(
                f"[Source: {source_name}{page_info}]\n{doc.page_content}"
            )
        context = "\n\n".join(context_parts)

        # Build messages
        messages = [
            {"role": "system", "content": build_system_prompt(config, context)}
        ]

        # Add conversation history (last 6 turns)
        for item in history[-6:]:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role and content is not None:
                    messages.append({"role": role, "content": content})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                human, assistant = item
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})

        messages.append({"role": "user", "content": question})

        # Call Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
        )

        answer = response.choices[0].message.content or ""
        # Detect and strip the unanswered marker (language-agnostic signal).
        answered = True
        stripped = answer.lstrip()
        if stripped.startswith(UNANSWERED_MARKER):
            answered = False
            answer = stripped[len(UNANSWERED_MARKER):].lstrip()
        logger.info("Answered question: %s...", question[:60])
        return answer, answered

    except Exception as e:
        error_str = str(e)
        if "401" in error_str:
            logger.error("Invalid API key")
            return "❌ Invalid API key. Please check your .env file.", True
        elif "429" in error_str:
            logger.warning("Rate limit hit")
            return "⏳ Rate limit reached. Please wait a moment.", True
        elif "503" in error_str:
            logger.warning("Groq service unavailable")
            return "🔄 AI service temporarily unavailable. Please try again.", True
        else:
            logger.error("Chat error: %s", e)
            return f"❌ Something went wrong: {error_str}", True


# ======================
# Chat History
# ======================

def save_history(history: list, filepath: str) -> None:
    """Save conversation history to a JSON file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save history to %s: %s", filepath, e)


def load_history(filepath: str) -> list:
    """Load conversation history from a JSON file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load history from %s: %s", filepath, e)
    return []


def export_history(history: list, prefix: str = "export") -> str:
    """
    Export chat history to TXT and JSON files.
    Returns a status message.
    """
    if not history:
        return "⚠️ No conversation history to export!"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exported = []

    # Export TXT
    txt_file = f"{prefix}_{timestamp}.txt"
    try:
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"Chat History Export\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            for i, item in enumerate(history, 1):
                if isinstance(item, dict):
                    role = item.get("role", "unknown")
                    content = item.get("content", "")
                    f.write(f"[{i}] {role.capitalize()}: {content}\n\n")
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    q, a = item
                    f.write(f"[{i}] Customer: {q}\n    Agent: {a}\n\n")
        exported.append(f"📄 {txt_file}")
    except Exception as e:
        exported.append(f"❌ TXT failed: {e}")

    # Export JSON
    json_file = f"{prefix}_{timestamp}.json"
    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_messages": len(history),
                "conversations": history,
            }, f, indent=2, ensure_ascii=False)
        exported.append(f"📊 {json_file}")
    except Exception as e:
        exported.append(f"❌ JSON failed: {e}")

    return "✅ Exported successfully!\n" + "\n".join(exported)
