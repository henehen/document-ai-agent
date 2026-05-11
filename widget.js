// ============================================
// Document AI Agent - Chat Widget v2.0
// Embed on any website with one script tag
// Mobile responsive, accessible, markdown support
// ============================================

(function() {
    const SERVER_URL = window.location.origin;
    const SESSION_ID = Math.random().toString(36).slice(2, 11);
    
    let config = {
        agent_name: "Agent",
        company_name: "Business",
        welcome_message: "Hello! How can I help you today?"
    };
    let isOpen = false;
    let isLoading = false;

    // Fetch config from server
    fetch(`${SERVER_URL}/config`)
        .then(r => r.json())
        .then(c => {
            config = c;
            updateWidget();
        })
        .catch(() => updateWidget());

    // Create styles
    const style = document.createElement('style');
    style.textContent = `
        #ai-widget-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            cursor: pointer;
            font-size: 24px;
            box-shadow: 0 4px 15px rgba(102,126,234,0.4);
            z-index: 99999;
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #ai-widget-btn:hover { 
            transform: scale(1.1); 
            box-shadow: 0 6px 25px rgba(102,126,234,0.6);
        }
        #ai-widget-btn:focus {
            outline: 3px solid rgba(102,126,234,0.8);
            outline-offset: 2px;
        }
        
        #ai-widget-container {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 380px;
            max-width: calc(100vw - 40px);
            height: 520px;
            max-height: calc(100vh - 120px);
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            display: none;
            flex-direction: column;
            z-index: 99999;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            animation: ai-widget-slide-up 0.3s ease-out;
        }

        @keyframes ai-widget-slide-up {
            from { 
                opacity: 0; 
                transform: translateY(20px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }

        /* Mobile: full width at bottom */
        @media (max-width: 480px) {
            #ai-widget-container {
                bottom: 0;
                right: 0;
                width: 100vw;
                max-width: 100vw;
                height: 85vh;
                max-height: 85vh;
                border-radius: 16px 16px 0 0;
            }
            #ai-widget-btn {
                bottom: 16px;
                right: 16px;
                width: 56px;
                height: 56px;
            }
        }
        
        #ai-widget-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }
        
        #ai-widget-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }
        
        #ai-widget-header p {
            margin: 2px 0 0 0;
            font-size: 12px;
            opacity: 0.8;
        }
        
        #ai-widget-close {
            background: none;
            border: none;
            color: white;
            font-size: 22px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 6px;
            transition: background 0.15s;
        }
        #ai-widget-close:hover {
            background: rgba(255,255,255,0.15);
        }
        #ai-widget-close:focus {
            outline: 2px solid rgba(255,255,255,0.5);
        }
        
        #ai-widget-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            scroll-behavior: smooth;
        }
        
        .ai-message {
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 16px;
            font-size: 14px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        
        .ai-message.bot {
            background: #f0f2f5;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            color: #1a1a2e;
        }

        .ai-message.bot strong { font-weight: 600; }
        .ai-message.bot code {
            background: #e2e5ea;
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 13px;
        }
        
        .ai-message.user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }

        .ai-message .timestamp {
            font-size: 10px;
            opacity: 0.6;
            margin-top: 4px;
            display: block;
        }
        
        .ai-message.typing {
            background: #f0f2f5;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            padding: 12px 18px;
        }

        .typing-dots {
            display: flex;
            gap: 4px;
        }
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: ai-dot-bounce 1.4s infinite ease-in-out;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes ai-dot-bounce {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        #ai-widget-input-area {
            padding: 12px 16px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
            flex-shrink: 0;
            background: white;
        }
        
        #ai-widget-input {
            flex: 1;
            border: 1px solid #ddd;
            border-radius: 24px;
            padding: 10px 16px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
            font-family: inherit;
        }
        
        #ai-widget-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102,126,234,0.15);
        }

        #ai-widget-input:disabled {
            background: #f5f5f5;
        }
        
        #ai-widget-send {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: opacity 0.2s, transform 0.15s;
            flex-shrink: 0;
        }
        #ai-widget-send:hover { transform: scale(1.05); }
        #ai-widget-send:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        #ai-widget-send:focus { outline: 3px solid rgba(102,126,234,0.5); }

        .ai-widget-powered {
            text-align: center;
            padding: 6px;
            font-size: 10px;
            color: #bbb;
            background: #fafafa;
            flex-shrink: 0;
        }
    `;
    document.head.appendChild(style);

    // Create widget HTML
    const container = document.createElement('div');
    container.id = 'ai-widget-container';
    container.setAttribute('role', 'dialog');
    container.setAttribute('aria-label', 'Chat with AI assistant');
    container.innerHTML = `
        <div id="ai-widget-header">
            <div>
                <h3 id="ai-widget-name">🤖 Agent</h3>
                <p id="ai-widget-company">Loading...</p>
            </div>
            <button id="ai-widget-close" aria-label="Close chat">×</button>
        </div>
        <div id="ai-widget-messages" role="log" aria-live="polite" aria-label="Chat messages"></div>
        <div id="ai-widget-input-area">
            <input id="ai-widget-input" type="text" placeholder="Type your message..." aria-label="Type your message" autocomplete="off">
            <button id="ai-widget-send" aria-label="Send message">➤</button>
        </div>
        <div class="ai-widget-powered">Powered by Document AI Agent</div>
    `;

    const button = document.createElement('button');
    button.id = 'ai-widget-btn';
    button.innerHTML = '💬';
    button.setAttribute('aria-label', 'Open chat');

    document.body.appendChild(container);
    document.body.appendChild(button);

    function updateWidget() {
        document.getElementById('ai-widget-name').textContent = `🤖 ${config.agent_name}`;
        document.getElementById('ai-widget-company').textContent = config.company_name;
        addMessage(config.welcome_message, 'bot');
    }

    function getTimeString() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function formatMessage(text) {
        // Escape HTML first to prevent XSS, then apply basic markdown-like formatting
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    function addMessage(text, type) {
        const messages = document.getElementById('ai-widget-messages');
        const msg = document.createElement('div');
        msg.className = `ai-message ${type}`;

        if (type === 'bot') {
            msg.innerHTML = formatMessage(text) + `<span class="timestamp">${getTimeString()}</span>`;
        } else {
            msg.textContent = text;
            const ts = document.createElement('span');
            ts.className = 'timestamp';
            ts.textContent = getTimeString();
            msg.appendChild(ts);
        }

        messages.appendChild(msg);
        messages.scrollTop = messages.scrollHeight;
        return msg;
    }

    function showTyping() {
        const messages = document.getElementById('ai-widget-messages');
        const msg = document.createElement('div');
        msg.className = 'ai-message typing';
        msg.id = 'ai-typing-indicator';
        msg.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
        messages.appendChild(msg);
        messages.scrollTop = messages.scrollHeight;
        return msg;
    }

    async function sendMessage() {
        if (isLoading) return;
        const input = document.getElementById('ai-widget-input');
        const sendBtn = document.getElementById('ai-widget-send');
        const message = input.value.trim();
        if (!message) return;

        isLoading = true;
        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;
        addMessage(message, 'user');

        const typing = showTyping();

        try {
            const formData = new FormData();
            formData.append('message', message);
            formData.append('session_id', SESSION_ID);

            const response = await fetch(`${SERVER_URL}/chat`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            typing.remove();
            addMessage(data.answer, 'bot');

        } catch (error) {
            typing.remove();
            addMessage('Sorry, something went wrong. Please try again.', 'bot');
        }

        isLoading = false;
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }

    // Event listeners
    button.addEventListener('click', () => {
        isOpen = !isOpen;
        container.style.display = isOpen ? 'flex' : 'none';
        button.innerHTML = isOpen ? '×' : '💬';
        button.setAttribute('aria-label', isOpen ? 'Close chat' : 'Open chat');
        if (isOpen) {
            document.getElementById('ai-widget-input').focus();
        }
    });

    document.getElementById('ai-widget-close').addEventListener('click', () => {
        isOpen = false;
        container.style.display = 'none';
        button.innerHTML = '💬';
        button.setAttribute('aria-label', 'Open chat');
    });

    document.getElementById('ai-widget-send').addEventListener('click', sendMessage);

    document.getElementById('ai-widget-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) sendMessage();
    });

    // Close with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isOpen) {
            isOpen = false;
            container.style.display = 'none';
            button.innerHTML = '💬';
            button.setAttribute('aria-label', 'Open chat');
            button.focus();
        }
    });
})();