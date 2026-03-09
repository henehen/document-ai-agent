// ============================================
// Document AI Agent - Chat Widget v1.3
// Embed on any website with one script tag
// ============================================

(function() {
    const SERVER_URL = window.location.origin;
    const SESSION_ID = Math.random().toString(36).substr(2, 9);
    
    let config = {
        agent_name: "Agent",
        company_name: "Business",
        welcome_message: "Hello! How can I help you today?"
    };

    // Fetch config from server
    fetch(`${SERVER_URL}/config`)
        .then(r => r.json())
        .then(c => {
            config = c;
            updateWidget();
        });

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
            z-index: 9999;
            transition: transform 0.2s;
        }
        #ai-widget-btn:hover { transform: scale(1.1); }
        
        #ai-widget-container {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            display: none;
            flex-direction: column;
            z-index: 9999;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }
        
        #ai-widget-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        #ai-widget-header h3 {
            margin: 0;
            font-size: 16px;
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
            font-size: 20px;
            cursor: pointer;
            padding: 0;
        }
        
        #ai-widget-messages {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .ai-message {
            max-width: 80%;
            padding: 10px 14px;
            border-radius: 15px;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .ai-message.bot {
            background: #f0f0f0;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        
        .ai-message.user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        
        .ai-message.typing {
            background: #f0f0f0;
            align-self: flex-start;
            color: #999;
            font-style: italic;
        }
        
        #ai-widget-input-area {
            padding: 15px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }
        
        #ai-widget-input {
            flex: 1;
            border: 1px solid #ddd;
            border-radius: 20px;
            padding: 8px 15px;
            font-size: 14px;
            outline: none;
        }
        
        #ai-widget-input:focus {
            border-color: #667eea;
        }
        
        #ai-widget-send {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    `;
    document.head.appendChild(style);

    // Create widget HTML
    const container = document.createElement('div');
    container.id = 'ai-widget-container';
    container.innerHTML = `
        <div id="ai-widget-header">
            <div>
                <h3 id="ai-widget-name">🤖 Agent</h3>
                <p id="ai-widget-company">Loading...</p>
            </div>
            <button id="ai-widget-close">×</button>
        </div>
        <div id="ai-widget-messages"></div>
        <div id="ai-widget-input-area">
            <input id="ai-widget-input" type="text" placeholder="Type your message...">
            <button id="ai-widget-send">➤</button>
        </div>
    `;

    const button = document.createElement('button');
    button.id = 'ai-widget-btn';
    button.innerHTML = '💬';

    document.body.appendChild(container);
    document.body.appendChild(button);

    function updateWidget() {
        document.getElementById('ai-widget-name').textContent = `🤖 ${config.agent_name}`;
        document.getElementById('ai-widget-company').textContent = config.company_name;
        addMessage(config.welcome_message, 'bot');
    }

    function addMessage(text, type) {
        const messages = document.getElementById('ai-widget-messages');
        const msg = document.createElement('div');
        msg.className = `ai-message ${type}`;
        msg.textContent = text;
        messages.appendChild(msg);
        messages.scrollTop = messages.scrollHeight;
        return msg;
    }

    async function sendMessage() {
        const input = document.getElementById('ai-widget-input');
        const message = input.value.trim();
        if (!message) return;

        input.value = '';
        addMessage(message, 'user');

        const typing = addMessage('Thinking...', 'typing');

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
    }

    // Event listeners
    button.addEventListener('click', () => {
        const isOpen = container.style.display === 'flex';
        container.style.display = isOpen ? 'none' : 'flex';
        button.innerHTML = isOpen ? '💬' : '×';
    });

    document.getElementById('ai-widget-close').addEventListener('click', () => {
        container.style.display = 'none';
        button.innerHTML = '💬';
    });

    document.getElementById('ai-widget-send').addEventListener('click', sendMessage);

    document.getElementById('ai-widget-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
})();