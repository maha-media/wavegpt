import { useState, useRef, useEffect } from 'react'
import { useChat } from '../hooks/useChat'

export function ChatPanel() {
  const { messages, sendMessage, isStreaming, clearChat } = useChat()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isStreaming) return
    sendMessage(input.trim())
    setInput('')
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span>AI Assistant</span>
        {messages.length > 0 && (
          <button className="chat-clear" onClick={clearChat} title="Clear chat">
            Clear
          </button>
        )}
      </div>
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-empty">Ask about your portfolio...</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg chat-${m.role}`}>
            {m.content}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={isStreaming ? 'Thinking...' : 'Message...'}
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
