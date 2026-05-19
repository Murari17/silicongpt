document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("chat-form");
    const input = document.getElementById("user-input");
    const history = document.getElementById("chat-history");
    const sendBtn = document.getElementById("send-btn");

    // Fetch Model Info
    fetch("/api/info")
        .then(res => res.json())
        .then(data => {
            const infoBox = document.getElementById("model-info");
            infoBox.innerHTML = "";
            for (const [key, value] of Object.entries(data)) {
                infoBox.innerHTML += `
                    <div class="info-item">
                        <span class="info-label">${key}</span>
                        <span class="info-value">${value}</span>
                    </div>
                `;
            }
        })
        .catch(err => {
            document.getElementById("model-info").innerHTML = "Error loading model info.";
        });

    const userSvg = `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
    const aiSvg = `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>`;

    function addMessage(content, isUser, source = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${isUser ? 'user-message' : 'ai-message'} fade-in`;
        
        let sourceHtml = "";
        if (!isUser && source) {
            const sourceClass = source.toLowerCase().includes("retrieved") ? "source-retrieved" : "source-generated";
            sourceHtml = `<div class="source-tag ${sourceClass}">${source}</div>`;
        }

        msgDiv.innerHTML = `
            <div class="avatar ${isUser ? 'user-avatar' : 'ai-avatar'}">${isUser ? userSvg : aiSvg}</div>
            <div class="message-content">
                <p></p>
                ${sourceHtml}
            </div>
        `;
        history.appendChild(msgDiv);
        history.scrollTop = history.scrollHeight;

        return msgDiv.querySelector("p");
    }

    function addTypingIndicator() {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ai-message fade-in typing-indicator-container`;
        msgDiv.innerHTML = `
            <div class="avatar ai-avatar">${aiSvg}</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        history.appendChild(msgDiv);
        history.scrollTop = history.scrollHeight;
        return msgDiv;
    }

    // Typewriter effect
    async function typeText(element, text) {
        element.textContent = "";
        for (let i = 0; i < text.length; i++) {
            element.textContent += text.charAt(i);
            history.scrollTop = history.scrollHeight;
            await new Promise(r => setTimeout(r, 10)); // 10ms per char
        }
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;

        input.value = "";
        input.disabled = true;
        sendBtn.disabled = true;

        // User message
        const userP = addMessage(text, true);
        userP.textContent = text;

        // Typing indicator
        const typingMsg = addTypingIndicator();

        try {
            const reqData = {
                message: text,
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                max_tokens: 128,
                rep_penalty: 1.15
            };

            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(reqData)
            });

            const data = await response.json();
            
            // Remove typing indicator
            typingMsg.remove();

            if (!response.ok) throw new Error(data.detail || "Server error");

            // Add final message and animate text
            const p = addMessage(data.answer, false, data.source);
            await typeText(p, data.answer);

        } catch (error) {
            typingMsg.remove();
            const p = addMessage("Connection error. Is the server running?", false, "Error");
            p.textContent = error.message;
        } finally {
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
        }
    });
});
