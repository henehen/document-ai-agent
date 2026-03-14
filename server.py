# ============================================
# Document AI Agent - Widget Server v1.3
# Author: Henrique Faria Cl
# Description: FastAPI server that powers
#              the embeddable chat widget
# ============================================

import os
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import shutil

# ---- LOAD API KEYS ----
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- LOAD CONFIG ----
CONFIG_FILE = "config.json"

def load_config():
    default = {
        "agent_name": "Sarah",
        "company_name": "My Business",
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

# ---- GLOBALS ----
retriever = None
client = None
chat_histories = {}

# ---- FASTAPI APP ----
app = FastAPI(title="Document AI Agent Widget")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- ROUTES ----

@app.get("/")
async def home():
    """Demo page showing the widget"""
    config = load_config()
    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>{config['agent_name']} - {config['company_name']}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; }}
        .demo-box {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        code {{
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            display: block;
            margin: 10px 0;
            font-size: 14px;
        }}
        .status {{
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .ok {{ background: #d4edda; color: #155724; }}
        .error {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="demo-box">
        <h1>🤖 {config['agent_name']} Widget Demo</h1>
        <h2>{config['company_name']}</h2>
        
        <h3>📋 Setup Instructions</h3>
        <p>1. Upload your business documents:</p>
        <p>2. Add this code to any website:</p>
        <code>&lt;script src="http://YOUR_SERVER/widget.js"&gt;&lt;/script&gt;</code>
        
        <div class="status ok">
            ✅ Server is running!
        </div>
        
        <p>The chat bubble should appear in the bottom-right corner of this page.</p>
    </div>
    <script src="/widget.js"></script>
</body>
</html>
""")

# ---- SIMPLE LIGHTWEIGHT RETRIEVER ----
class SimpleRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        texts = [c.page_content for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def invoke(self, query):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(scores)[-4:][::-1]
        return [self.chunks[i] for i in top_indices]
@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload and process business documents"""
    global retriever

    supported = {".pdf", ".docx", ".txt"}
    documents = []
    loaded = []
    failed = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in supported:
            failed.append(f"{file.filename}: unsupported format")
            continue

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name

            loaders = {
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".txt": TextLoader
            }

            loader = loaders[ext](tmp_path)
            documents.extend(loader.load())
            loaded.append(file.filename)
            os.unlink(tmp_path)

        except Exception as e:
            failed.append(f"{file.filename}: {str(e)}")

    if not documents:
        return {"success": False, "error": "No documents loaded"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    retriever = SimpleRetriever(chunks)
    return {
        "success": True,
        "loaded": loaded,
        "failed": failed,
        "message": f"✅ Loaded {len(loaded)} document(s)"
    }

@app.get("/debug-key")
async def debug_key():
    key = os.environ.get("GROQ_API_KEY", "NOT FOUND")
    return {
        "key_found": key != "NOT FOUND",
        "key_length": len(key),
        "key_start": key[:8] if len(key) > 8 else "too short"
    }
@app.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form(default="default")
):
    """Process customer message and return answer"""
    global retriever, chat_histories

    config = load_config()

    if retriever is None:
        return {"answer": "⚠️ No documents loaded yet. Please upload documents first."}

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]

    global client
    if client is None:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    try:
        relevant_docs = retriever.invoke(message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            {
                "role": "system",
                "content": f"""You are {config['agent_name']}, customer service for {config['company_name']}.
Tone: {config['tone']}

LANGUAGE RULE: Always respond in the same language the customer uses.

Business information:
{context}

Rules:
- Be {config['tone']} and helpful
- Keep answers concise
- If unknown: "{config['unknown_answer']}"
- Never make up information"""
            }
        ]

        for human, assistant in history[-6:]:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=300
        )

        answer = response.choices[0].message.content
        history.append((message, answer))
        chat_histories[session_id] = history

        return {"answer": answer}

    except Exception as e:
        if "401" in str(e):
            return {"answer": "❌ API key error. Please contact support."}
        elif "429" in str(e):
            return {"answer": "⏳ Too many requests. Please wait a moment."}
        else:
            return {"answer": f"❌ Error: {str(e)}"}

@app.get("/config")
async def get_config():
    """Return public config for widget"""
    config = load_config()
    return {
        "agent_name": config["agent_name"],
        "company_name": config["company_name"],
        "welcome_message": config["welcome_message"]
    }

@app.get("/widget.js")
async def widget_js():
    """Serve the widget JavaScript"""
    return FileResponse("widget.js", media_type="application/javascript")
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)