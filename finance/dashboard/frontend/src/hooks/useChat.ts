import { useState, useCallback } from 'react'
import type { ChatMessage } from '../types'

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const saved = sessionStorage.getItem('chat_history')
    return saved ? JSON.parse(saved) : []
  })
  const [isStreaming, setIsStreaming] = useState(false)

  const sendMessage = useCallback(async (content: string) => {
    const userMsg: ChatMessage = { role: 'user', content }
    setMessages(prev => {
      const updated = [...prev, userMsg]
      sessionStorage.setItem('chat_history', JSON.stringify(updated))
      return updated
    })

    setIsStreaming(true)
    let assistantText = ''

    try {
      const response = await fetch('/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMsg],
        }),
      })

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.text) {
                assistantText += data.text
                setMessages(prev => {
                  const next = [...prev]
                  const last = next[next.length - 1]
                  if (last?.role === 'assistant') {
                    return [...next.slice(0, -1), { role: 'assistant', content: assistantText }]
                  }
                  return [...next, { role: 'assistant', content: assistantText }]
                })
              }
            } catch { /* ignore */ }
          }
        }
      }
    } finally {
      setIsStreaming(false)
      setMessages(prev => {
        sessionStorage.setItem('chat_history', JSON.stringify(prev))
        return prev
      })
    }
  }, [messages])

  const clearChat = useCallback(() => {
    setMessages([])
    sessionStorage.removeItem('chat_history')
  }, [])

  return { messages, sendMessage, isStreaming, clearChat }
}
