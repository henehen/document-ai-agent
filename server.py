# ============================================
# Document AI Agent - Widget Server v2.0
# Author: Henrique Faria
# Description: FastAPI server with analytics,
#              email alerts, admin panel,
#              conversation logs
# ============================================

import os
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
        "unknown_answer": "I'm sorry, I don't have that information. Please contact our support team.",
        "admin_password": "changeme123"
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

# ---- ANALYTICS ----
analytics = {
    "total_questions": 0,
    "unanswered_questions": 0,
    "languages_detected": {},
    "questions_per_day": {},
    "recent_questions": []
}

# ---- CONVERSATION LOGS ----
conversation_logs = []

def log_conversation(session_id, question, answer, language="unknown"):
    """Log every conversation"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "language": language
    }
    conversation_logs.append(entry)

    # Update analytics
    analytics["total_questions"] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    analytics["questions_per_day"][today] = analytics["questions_per_day"].get(today, 0) + 1
    analytics["recent_questions"].append({
        "timestamp": entry["timestamp"],
        "question": question[:100]
    })
    if len(analytics["recent_questions"]) > 20:
        analytics["recent_questions"].pop(0)

def detect_language(text):
    """Simple language detection"""
    french_words = ["bonjour", "merci", "comment", "quelles", "votre", "est", "les", "des"]
    spanish_words = ["hola", "gracias", "como", "cuales", "tiene", "que", "los", "una"]
    portuguese_words = ["olá", "obrigado", "como", "quais", "tem", "que", "os", "uma"]

    text_lower = text.lower()
    if any(w in text_lower for w in french_words):
        return "French"
    elif any(w in text_lower for w in spanish_words):
        return "Spanish"
    elif any(w in text_lower for w in portuguese_words):
        return "Portuguese"
    return "English"

def send_email_alert(question, session_id):
    """Send email when AI can't answer"""
    config = load_config()
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not all([smtp_host, smtp_user, smtp_pass]):
        return  # Email not configured, skip silently

    try:
        msg = MIMEText(f"""
Your AI agent could not answer a customer question.

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session: {session_id}
Question: {question}

Please update your FAQ documents to include this answer.
        """)
        msg["Subject"] = f"[{config['company_name']}] AI Agent - Unanswered Question"
        msg["From"] = smtp_user
        msg["To"] = config["support_email"]

        with smtplib.SMTP_SSL(smtp_host, 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, config["support_email"], msg.as_string())
    except Exception:
        pass  # Don't crash if email fails

# ---- FASTAPI APP ----
app = FastAPI(title="Document AI Agent v2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- LIGHTWEIGHT RETRIEVER ----
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

# ---- PUBLIC ROUTES ----

@app.get("/")
async def home():
    config = load_config()
    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>{config['agent_name']} - {config['company_name']}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }}
        .demo-box {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        code {{ background: #f0f0f0; padding: 10px; border-radius: 5px; display: block; margin: 10px 0; font-size: 14px; }}
        .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .ok {{ background: #d4edda; color: #155724; }}
    </style>
</head>
<body>
    <div class="demo-box">
        <h1>🤖 {config['agent_name']} Widget Demo</h1>
        <h2>{config['company_name']}</h2>
        <h3>📋 Setup Instructions</h3>
        <p>Add this code to any website:</p>
        <code>&lt;script src="YOUR_SERVER_URL/widget.js"&gt;&lt;/script&gt;</code>
        <div class="status ok">✅ Server is running! v2.0</div>
        <p>The chat bubble should appear in the bottom-right corner.</p>
        <p><a href="/admin">🔐 Admin Panel</a></p>
    </div>
    <script src="/widget.js"></script>
</body>
</html>
""")

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
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
            loaders = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}
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
    return {"success": True, "loaded": loaded, "failed": failed, "message": f"✅ Loaded {len(loaded)} document(s)"}

@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form(default="default")):
    global retriever, client, chat_histories

    config = load_config()

    if retriever is None:
        return {"answer": "⚠️ No documents loaded yet."}

    global client
    if client is None:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]
    language = detect_language(message)
    analytics["languages_detected"][language] = analytics["languages_detected"].get(language, 0) + 1

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
- If unknown say exactly: "UNKNOWN_QUESTION: " followed by your polite response
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

        # Check if question was unanswered
        if "UNKNOWN_QUESTION:" in answer:
            analytics["unanswered_questions"] += 1
            answer = answer.replace("UNKNOWN_QUESTION: ", "")
            send_email_alert(message, session_id)

        history.append((message, answer))
        chat_histories[session_id] = history
        log_conversation(session_id, message, answer, language)

        return {"answer": answer}

    except Exception as e:
        if "401" in str(e):
            return {"answer": f"❌ 401 Error: {str(e)}"}
        elif "429" in str(e):
            return {"answer": "⏳ Too many requests. Please wait a moment."}
        else:
            return {"answer": f"❌ Error: {str(e)}"}

@app.get("/config")
async def get_config():
    config = load_config()
    return {
        "agent_name": config["agent_name"],
        "company_name": config["company_name"],
        "welcome_message": config["welcome_message"]
    }

@app.get("/widget.js")
async def widget_js():
    return FileResponse("widget.js", media_type="application/javascript")

@app.get("/debug-key")
async def debug_key():
    key = os.environ.get("GROQ_API_KEY", "NOT FOUND")
    return {"key_found": key != "NOT FOUND", "key_length": len(key), "key_start": key[:8] if len(key) > 8 else "too short"}

# ---- ADMIN ROUTES ----

def check_admin(password: str):
    config = load_config()
    if password != config.get("admin_password", "changeme123"):
        raise HTTPException(status_code=401, detail="Wrong password")

@app.get("/admin")
async def admin_panel():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel - Document AI Agent v2.0</title>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px 30px; }
        .header h1 { color: white; font-size: 24px; }
        .header p { color: rgba(255,255,255,0.8); font-size: 14px; }
        .container { max-width: 1200px; margin: 30px auto; padding: 0 20px; }
        .login-box { background: #16213e; padding: 40px; border-radius: 15px; max-width: 400px; margin: 100px auto; text-align: center; }
        .login-box h2 { margin-bottom: 20px; color: #667eea; }
        input[type=password] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #667eea; background: #0f3460; color: white; font-size: 16px; margin-bottom: 15px; }
        .btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%; }
        .btn:hover { opacity: 0.9; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #16213e; padding: 25px; border-radius: 15px; border-left: 4px solid #667eea; }
        .card h3 { color: #667eea; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; }
        .card .number { font-size: 48px; font-weight: bold; color: white; }
        .card .label { color: #aaa; font-size: 12px; }
        .section { background: #16213e; padding: 25px; border-radius: 15px; margin-bottom: 20px; }
        .section h2 { color: #667eea; margin-bottom: 20px; font-size: 18px; }
        .upload-area { border: 2px dashed #667eea; border-radius: 10px; padding: 30px; text-align: center; margin-bottom: 15px; }
        .log-entry { background: #0f3460; padding: 12px; border-radius: 8px; margin-bottom: 10px; font-size: 13px; }
        .log-entry .time { color: #667eea; font-size: 11px; }
        .log-entry .q { color: #eee; margin: 5px 0; }
        .log-entry .a { color: #aaa; }
        .lang-bar { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
        .lang-tag { background: #667eea; padding: 5px 12px; border-radius: 20px; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #0f3460; padding: 10px; text-align: left; color: #667eea; font-size: 13px; }
        td { padding: 10px; border-bottom: 1px solid #0f3460; font-size: 13px; color: #ccc; }
        #dashboard { display: none; }
    </style>
</head>
<body>

<div class="header">
    <h1>🤖 Document AI Agent v2.0</h1>
    <p>Admin Panel</p>
</div>

<!-- LOGIN -->
<div id="loginBox" class="login-box">
    <h2>🔐 Admin Login</h2>
    <input type="password" id="passwordInput" placeholder="Enter admin password" onkeypress="if(event.key==='Enter') login()">
    <button class="btn" onclick="login()">Login</button>
    <p id="loginError" style="color:#ff6b6b;margin-top:10px;"></p>
</div>

<!-- DASHBOARD -->
<div id="dashboard">
<div class="container">

    <!-- STATS -->
    <div class="grid" id="statsGrid">
        <div class="card">
            <h3>Total Questions</h3>
            <div class="number" id="totalQ">0</div>
            <div class="label">All time</div>
        </div>
        <div class="card">
            <h3>Unanswered</h3>
            <div class="number" id="unansweredQ">0</div>
            <div class="label">Need attention</div>
        </div>
        <div class="card">
            <h3>Today</h3>
            <div class="number" id="todayQ">0</div>
            <div class="label">Questions today</div>
        </div>
        <div class="card">
            <h3>Languages</h3>
            <div class="lang-bar" id="langBar"></div>
        </div>
    </div>

    <!-- UPLOAD DOCUMENTS -->
    <div class="section">
        <h2>📂 Upload Documents</h2>
        <div class="upload-area">
            <p style="margin-bottom:15px;">📄 Upload PDF, DOCX or TXT files</p>
            <input type="file" id="fileInput" multiple accept=".pdf,.docx,.txt" style="margin-bottom:15px;">
            <br>
            <button class="btn" onclick="uploadDocs()" style="width:auto;padding:10px 25px;">🚀 Upload & Train AI</button>
        </div>
        <p id="uploadStatus" style="color:#4caf50;"></p>
    </div>

    <!-- CONVERSATION LOGS -->
    <div class="section">
        <h2>💬 Recent Conversations</h2>
        <div id="logsContainer">
            <p style="color:#aaa;">No conversations yet.</p>
        </div>
    </div>

    <!-- QUESTIONS PER DAY -->
    <div class="section">
        <h2>📊 Questions Per Day</h2>
        <table>
            <tr><th>Date</th><th>Questions</th></tr>
            <tbody id="dailyTable"></tbody>
        </table>
    </div>

</div>
</div>

<script>
let adminPassword = '';

async function login() {
    const pwd = document.getElementById('passwordInput').value;
    const res = await fetch('/admin/verify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({password: pwd})
    });
    const data = await res.json();
    if (data.success) {
        adminPassword = pwd;
        document.getElementById('loginBox').style.display = 'none';
        document.getElementById('dashboard').style.display = 'block';
        loadDashboard();
    } else {
        document.getElementById('loginError').textContent = '❌ Wrong password!';
    }
}

async function loadDashboard() {
    const res = await fetch('/admin/data?password=' + adminPassword);
    const data = await res.json();

    document.getElementById('totalQ').textContent = data.analytics.total_questions;
    document.getElementById('unansweredQ').textContent = data.analytics.unanswered_questions;

    const today = new Date().toISOString().split('T')[0];
    document.getElementById('todayQ').textContent = data.analytics.questions_per_day[today] || 0;

    const langBar = document.getElementById('langBar');
    langBar.innerHTML = '';
    for (const [lang, count] of Object.entries(data.analytics.languages_detected)) {
        langBar.innerHTML += `<span class="lang-tag">${lang}: ${count}</span>`;
    }

    const logs = document.getElementById('logsContainer');
    if (data.logs.length === 0) {
        logs.innerHTML = '<p style="color:#aaa;">No conversations yet.</p>';
    } else {
        logs.innerHTML = data.logs.slice(-10).reverse().map(l => `
            <div class="log-entry">
                <div class="time">${l.timestamp} | ${l.language} | Session: ${l.session_id}</div>
                <div class="q">❓ ${l.question}</div>
                <div class="a">🤖 ${l.answer}</div>
            </div>
        `).join('');
    }

    const daily = document.getElementById('dailyTable');
    daily.innerHTML = Object.entries(data.analytics.questions_per_day)
        .reverse()
        .map(([date, count]) => `<tr><td>${date}</td><td>${count}</td></tr>`)
        .join('');

    setTimeout(loadDashboard, 30000);
}

async function uploadDocs() {
    const files = document.getElementById('fileInput').files;
    if (!files.length) { alert('Please select files first!'); return; }

    const formData = new FormData();
    for (const file of files) formData.append('files', file);

    document.getElementById('uploadStatus').textContent = '⏳ Uploading...';
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    document.getElementById('uploadStatus').textContent = data.message || data.error;
}
</script>

</body>
</html>
""")

@app.post("/admin/verify")
async def admin_verify(request: Request):
    body = await request.json()
    config = load_config()
    if body.get("password") == config.get("admin_password", "changeme123"):
        return {"success": True}
    return {"success": False}

@app.get("/admin/data")
async def admin_data(password: str):
    check_admin(password)
    return {
        "analytics": analytics,
        "logs": conversation_logs
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)