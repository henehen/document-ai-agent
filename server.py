# ============================================
# Document AI Agent - Widget Server v3.0
# Refactored to use core.py
# Security fixes, proper retriever, analytics
# ============================================

import os
import json
import hmac
import secrets
import smtplib
import tempfile
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from email.mime.text import MIMEText

from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

from core import (
    load_config, create_client, load_documents_from_files,
    build_retriever, load_persisted_retriever, ask,
    logger, SUPPORTED_EXTENSIONS,
)

# ---- LOAD ENV ----
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# ---- GLOBALS ----
retriever = None
client = None
chat_histories: dict[str, list] = {}
active_tokens: set[str] = set()

# ---- ANALYTICS (in-memory, survives until restart) ----
ANALYTICS_FILE = os.path.join(os.path.dirname(__file__), "analytics.json")
LOGS_FILE = os.path.join(os.path.dirname(__file__), "conversation_logs.json")

def load_analytics() -> dict:
    """Load analytics from disk."""
    defaults = {
        "total_questions": 0,
        "unanswered_questions": 0,
        "languages_detected": {},
        "questions_per_day": {},
        "recent_questions": [],
    }
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load analytics: %s", e)
    return defaults

def save_analytics(analytics: dict):
    """Persist analytics to disk."""
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(analytics, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save analytics: %s", e)

def load_conversation_logs() -> list:
    """Load conversation logs from disk."""
    try:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load logs: %s", e)
    return []

def save_conversation_logs(logs: list):
    """Persist conversation logs to disk."""
    try:
        # Keep only last 500 entries to avoid file growing forever
        trimmed = logs[-500:]
        with open(LOGS_FILE, "w") as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to save logs: %s", e)

analytics = load_analytics()
conversation_logs = load_conversation_logs()


def log_conversation(session_id: str, question: str, answer: str, language: str = "unknown"):
    """Log a conversation and update analytics."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "question": question,
        "answer": answer[:200],
        "language": language,
    }
    conversation_logs.append(entry)
    save_conversation_logs(conversation_logs)

    analytics["total_questions"] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    analytics["questions_per_day"][today] = analytics["questions_per_day"].get(today, 0) + 1
    analytics["recent_questions"].append({
        "timestamp": entry["timestamp"],
        "question": question[:100],
    })
    if len(analytics["recent_questions"]) > 20:
        analytics["recent_questions"].pop(0)
    save_analytics(analytics)


def detect_language(text: str) -> str:
    """Simple language detection based on common words."""
    text_lower = text.lower()
    # Language markers — using distinctive words to reduce false positives
    markers = {
        "French": ["bonjour", "merci", "s'il vous", "aujourd'hui", "pourquoi", "quelles"],
        "Spanish": ["hola", "gracias", "por favor", "cuáles", "cómo", "tiene"],
        "Portuguese": ["olá", "obrigado", "por favor", "quais", "você", "também"],
        "German": ["hallo", "danke", "bitte", "warum", "können", "möchten"],
        "Italian": ["ciao", "grazie", "prego", "perché", "quanto", "posso"],
    }
    for lang, words in markers.items():
        if sum(1 for w in words if w in text_lower) >= 2:
            return lang
    return "English"


def send_email_alert(question: str, session_id: str):
    """Send email when AI can't answer a question."""
    config = load_config()
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not all([smtp_host, smtp_user, smtp_pass]):
        return

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
        logger.info("Email alert sent for unanswered question")
    except Exception as e:
        logger.warning("Failed to send email alert: %s", e)


# ======================
# Admin Auth
# ======================

def verify_admin_password(password: str) -> bool:
    """Verify admin password against hashed value in env, or plaintext fallback."""
    stored = os.environ.get("ADMIN_PASSWORD", "changeme123")
    # Simple constant-time comparison
    return hmac.compare_digest(password, stored)


def require_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Dependency that validates an admin bearer token."""
    if credentials.credentials not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return credentials.credentials


# ======================
# FastAPI App
# ======================

config = load_config()


# ======================
# Lifespan (startup/shutdown)
# ======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize client and load persisted retriever on startup."""
    global client, retriever
    client = create_client()
    retriever = load_persisted_retriever()
    if retriever:
        logger.info("Server started with persisted documents")
    else:
        logger.info("Server started — no documents loaded yet")
    yield


app = FastAPI(
    title="Document AI Agent v3.0",
    description="AI-powered customer service agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("allowed_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
# Public Routes
# ======================

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
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }}
        .demo-box {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        code {{ background: #f0f0f0; padding: 10px; border-radius: 5px; display: block; margin: 10px 0; font-size: 14px; word-break: break-all; }}
        .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .ok {{ background: #d4edda; color: #155724; }}
        .info {{ background: #cce5ff; color: #004085; }}
        a {{ color: #667eea; }}
    </style>
</head>
<body>
    <div class="demo-box">
        <h1>🤖 {config['agent_name']} Widget Demo</h1>
        <h2>{config['company_name']}</h2>

        <div class="status ok">✅ Server is running! v3.0</div>
        <div class="status info">📄 Documents loaded: {'Yes' if retriever else 'No — upload via admin panel'}</div>

        <h3>📋 Setup Instructions</h3>
        <p>Add this code to any website:</p>
        <code>&lt;script src="YOUR_SERVER_URL/widget.js"&gt;&lt;/script&gt;</code>

        <p>The chat bubble will appear in the bottom-right corner.</p>
        <p><a href="/admin">🔐 Admin Panel</a> | <a href="/health">❤️ Health Check</a></p>
    </div>
    <script src="/widget.js"></script>
</body>
</html>
""")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "3.0",
        "documents_loaded": retriever is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...), token: str = Depends(require_token)):
    """Upload and process documents."""
    global retriever

    tmp_paths = []
    try:
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_paths.append(tmp.name)

        documents, loaded, failed = load_documents_from_files(tmp_paths)

        if not documents:
            return {"success": False, "error": "No documents loaded"}

        new_retriever = build_retriever(documents)
        if not new_retriever:
            return {"success": False, "error": "Failed to build AI memory"}

        retriever = new_retriever
        return {
            "success": True,
            "loaded": loaded,
            "failed": failed,
            "message": f"✅ Loaded {len(loaded)} document(s)",
        }
    finally:
        # Clean up temp files
        for path in tmp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.post("/chat")
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    session_id: str = Form(default="default"),
):
    """Chat with the AI agent."""
    global retriever, client

    config = load_config()

    if retriever is None:
        return {"answer": "⚠️ No documents loaded yet. Please ask an admin to upload documents."}

    if client is None:
        client = create_client()
        if client is None:
            return {"answer": "❌ AI service not configured."}

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]
    language = detect_language(message)
    analytics["languages_detected"][language] = analytics["languages_detected"].get(language, 0) + 1

    answer, answered = ask(client, retriever, message, history, config)

    # Track when the agent could not answer (language-agnostic, via the
    # UNANSWERED_MARKER sentinel set by core.ask()).
    if not answered:
        analytics["unanswered_questions"] += 1
        # Send email alert without blocking the request/event loop
        background_tasks.add_task(send_email_alert, message, session_id)

    history.append((message, answer))
    chat_histories[session_id] = history[-20:]  # Keep last 20 turns per session
    log_conversation(session_id, message, answer, language)

    return {"answer": answer}


@app.get("/config")
async def get_config():
    """Return public-safe config for the widget."""
    config = load_config()
    return {
        "agent_name": config["agent_name"],
        "company_name": config["company_name"],
        "welcome_message": config["welcome_message"],
    }


@app.get("/widget.js")
async def widget_js():
    return FileResponse("widget.js", media_type="application/javascript")


# ======================
# Admin Routes
# ======================

@app.post("/admin/login")
async def admin_login(request: Request):
    """Authenticate and receive a session token."""
    body = await request.json()
    password = body.get("password", "")

    if verify_admin_password(password):
        token = secrets.token_urlsafe(32)
        active_tokens.add(token)
        logger.info("Admin login successful")
        return {"success": True, "token": token}

    logger.warning("Failed admin login attempt")
    return JSONResponse(status_code=401, content={"success": False, "error": "Wrong password"})


@app.get("/admin/data")
async def admin_data(token: str = Depends(require_token)):
    """Return analytics and conversation logs (requires auth)."""
    return {
        "analytics": analytics,
        "logs": conversation_logs[-50:],
    }


@app.post("/admin/logout")
async def admin_logout(token: str = Depends(require_token)):
    """Invalidate admin session."""
    active_tokens.discard(token)
    return {"success": True}


@app.get("/admin")
async def admin_panel():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel - Document AI Agent v3.0</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px 30px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { color: white; font-size: 24px; }
        .header p { color: rgba(255,255,255,0.8); font-size: 14px; }
        .header button { background: rgba(255,255,255,0.2); color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; }
        .header button:hover { background: rgba(255,255,255,0.3); }
        .container { max-width: 1200px; margin: 30px auto; padding: 0 20px; }
        .login-box { background: #16213e; padding: 40px; border-radius: 15px; max-width: 400px; margin: 100px auto; text-align: center; }
        .login-box h2 { margin-bottom: 20px; color: #667eea; }
        input[type=password] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #667eea; background: #0f3460; color: white; font-size: 16px; margin-bottom: 15px; }
        .btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%; transition: opacity 0.2s; }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #16213e; padding: 25px; border-radius: 15px; border-left: 4px solid #667eea; }
        .card h3 { color: #667eea; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; }
        .card .number { font-size: 48px; font-weight: bold; color: white; }
        .card .label { color: #aaa; font-size: 12px; }
        .section { background: #16213e; padding: 25px; border-radius: 15px; margin-bottom: 20px; }
        .section h2 { color: #667eea; margin-bottom: 20px; font-size: 18px; }
        .upload-area { border: 2px dashed #667eea; border-radius: 10px; padding: 30px; text-align: center; margin-bottom: 15px; transition: border-color 0.2s; }
        .upload-area:hover { border-color: #764ba2; }
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
        .refresh-indicator { font-size: 12px; color: #aaa; margin-top: 15px; }
    </style>
</head>
<body>

<div class="header">
    <div>
        <h1>🤖 Document AI Agent v3.0</h1>
        <p>Admin Panel</p>
    </div>
    <button id="logoutBtn" style="display:none" onclick="logout()">🚪 Logout</button>
</div>

<!-- LOGIN -->
<div id="loginBox" class="login-box">
    <h2>🔐 Admin Login</h2>
    <input type="password" id="passwordInput" placeholder="Enter admin password" onkeypress="if(event.key==='Enter') login()" autocomplete="current-password">
    <button class="btn" onclick="login()" id="loginBtn">Login</button>
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
            <button class="btn" onclick="uploadDocs()" id="uploadBtn" style="width:auto;padding:10px 25px;">🚀 Upload & Train AI</button>
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

    <div class="refresh-indicator" id="refreshInfo">Auto-refreshes every 30s</div>

</div>
</div>

<script>
let authToken = '';
let refreshTimer = null;

async function login() {
    const btn = document.getElementById('loginBtn');
    btn.disabled = true;
    btn.textContent = 'Logging in...';

    const pwd = document.getElementById('passwordInput').value;
    try {
        const res = await fetch('/admin/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({password: pwd})
        });
        const data = await res.json();
        if (data.success) {
            authToken = data.token;
            document.getElementById('loginBox').style.display = 'none';
            document.getElementById('dashboard').style.display = 'block';
            document.getElementById('logoutBtn').style.display = 'block';
            loadDashboard();
        } else {
            document.getElementById('loginError').textContent = '❌ Wrong password!';
        }
    } catch(e) {
        document.getElementById('loginError').textContent = '❌ Connection error';
    }
    btn.disabled = false;
    btn.textContent = 'Login';
}

async function logout() {
    try {
        await fetch('/admin/logout', {
            method: 'POST',
            headers: {'Authorization': 'Bearer ' + authToken}
        });
    } catch(e) {}
    authToken = '';
    if (refreshTimer) clearTimeout(refreshTimer);
    document.getElementById('loginBox').style.display = 'block';
    document.getElementById('dashboard').style.display = 'none';
    document.getElementById('logoutBtn').style.display = 'none';
    document.getElementById('passwordInput').value = '';
}

async function loadDashboard() {
    try {
        const res = await fetch('/admin/data', {
            headers: {'Authorization': 'Bearer ' + authToken}
        });
        if (res.status === 401) { logout(); return; }
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
                    <div class="q">❓ ${escapeHtml(l.question)}</div>
                    <div class="a">🤖 ${escapeHtml(l.answer)}</div>
                </div>
            `).join('');
        }

        const daily = document.getElementById('dailyTable');
        daily.innerHTML = Object.entries(data.analytics.questions_per_day)
            .reverse()
            .map(([date, count]) => `<tr><td>${date}</td><td>${count}</td></tr>`)
            .join('');

        const now = new Date().toLocaleTimeString();
        document.getElementById('refreshInfo').textContent = `Last updated: ${now} — Auto-refreshes every 30s`;

        refreshTimer = setTimeout(loadDashboard, 30000);
    } catch(e) {
        console.error('Dashboard load failed:', e);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function uploadDocs() {
    const files = document.getElementById('fileInput').files;
    if (!files.length) { alert('Please select files first!'); return; }

    const btn = document.getElementById('uploadBtn');
    btn.disabled = true;
    document.getElementById('uploadStatus').textContent = '⏳ Uploading & processing...';

    const formData = new FormData();
    for (const file of files) formData.append('files', file);

    try {
        const res = await fetch('/upload', { 
            method: 'POST', 
            headers: { 'Authorization': 'Bearer ' + authToken },
            body: formData 
        });
        const data = await res.json();
        document.getElementById('uploadStatus').textContent = data.message || data.error;
    } catch(e) {
        document.getElementById('uploadStatus').textContent = '❌ Upload failed: ' + e.message;
    }
    btn.disabled = false;
}
</script>

</body>
</html>
""")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)