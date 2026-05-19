import { useState, useEffect, useRef } from 'react'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  
  const historyRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight
    }
  }, [messages, isTyping])

  const handleInput = (e) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }

  const sendQuery = async (queryText) => {
    if (!queryText.trim() || isTyping) return

    setInput("")
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    
    setMessages(prev => [...prev, { role: 'user', text: queryText }])
    setIsTyping(true)

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: queryText,
          temperature: 0.7,
          top_k: 40,
          top_p: 0.9,
          max_tokens: 128,
          rep_penalty: 1.15
        })
      })
      
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || "Server error")
      
      setIsTyping(false)
      
      setMessages(prev => [...prev, { 
        role: 'ai', 
        text: "", 
        fullText: data.answer,
        source: data.source,
        typingNow: true
      }])
      
    } catch (err) {
      setIsTyping(false)
      setMessages(prev => [...prev, { 
        role: 'ai', 
        text: "Connection error. Is the server running?", 
        source: "Error" 
      }])
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    sendQuery(input);
  }

  // Effect to handle typewriter animation
  useEffect(() => {
    const lastMsg = messages[messages.length - 1]
    if (lastMsg && lastMsg.typingNow && lastMsg.text.length < lastMsg.fullText.length) {
      const timeout = setTimeout(() => {
        setMessages(prev => {
          const newMessages = [...prev]
          const msg = newMessages[newMessages.length - 1]
          msg.text = msg.fullText.substring(0, msg.text.length + 1)
          if (msg.text.length === msg.fullText.length) {
            msg.typingNow = false
          }
          return newMessages
        })
      }, 10)
      return () => clearTimeout(timeout)
    }
  }, [messages])

  const userSvg = (
    <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
      <circle cx="12" cy="7" r="4"></circle>
    </svg>
  )

  const aiSvg = (
    <img src="/logo.png" alt="SiliconGPT" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '50%' }} />
  )

  const suggestions = [
    "What is hardness of water?",
    "Explain environment & human health",
    "What is an aquifer?",
    "What is a wave?"
  ];

  return (
    <div className="app-container">
      <main className="chat-container">
        <div className="chat-header">
          <h2>SiliconGPT Core 1.0</h2>
        </div>

        <div className="chat-history" ref={historyRef}>
          {messages.length === 0 ? (
            <div className="welcome-screen fade-in">
              <div className="welcome-logo">
                <img src="/logo.png" alt="SiliconGPT Logo" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '50%' }} />
              </div>
              <h1>How can I help you today?</h1>
              <div className="suggestions-grid">
                {suggestions.map((text, idx) => (
                  <button key={idx} className="suggestion-card" onClick={() => sendQuery(text)}>
                    <span>{text}</span>
                    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="5" y1="12" x2="19" y2="12"></line>
                      <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role === 'user' ? 'user-message' : 'ai-message'} fade-in`}>
                <div className="message-inner">
                  <div className="avatar">
                    {msg.role === 'user' ? userSvg : aiSvg}
                  </div>
                  <div className="message-content">
                    <p>{msg.text || (msg.role === 'ai' && !msg.typingNow ? msg.fullText : msg.text)}</p>
                    {msg.source && (
                      <div className={`source-tag ${msg.source.toLowerCase().includes("retrieved") ? "source-retrieved" : "source-generated"}`}>
                        <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px'}}>
                          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                          <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        {msg.source}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
          
          {isTyping && (
            <div className="message ai-message fade-in">
              <div className="message-inner">
                <div className="avatar">{aiSvg}</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="chat-input-container">
          <form onSubmit={handleSubmit} className="input-glass">
            <textarea 
              id="user-input"
              ref={textareaRef}
              value={input}
              onChange={handleInput}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask SiliconGPT..." 
              disabled={isTyping}
              required
              rows={1}
            />
            <button type="submit" id="send-btn" disabled={isTyping || !input.trim()}>
              <svg viewBox="0 0 24 24" fill="none" width="16" height="16">
                <path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </form>
          <div className="footer-disclaimer">
            SiliconGPT can make mistakes. Check important information.
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
