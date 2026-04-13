import { useEffect, useRef } from 'react'

type SSEHandler = (data: any) => void

export function useSSE(url: string, handlers: Record<string, SSEHandler>) {
  const handlersRef = useRef(handlers)
  handlersRef.current = handlers

  useEffect(() => {
    const es = new EventSource(url)

    for (const eventType of Object.keys(handlersRef.current)) {
      es.addEventListener(eventType, (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data)
          handlersRef.current[eventType]?.(data)
        } catch {
          // ignore parse errors
        }
      })
    }

    es.onerror = () => {
      console.warn('SSE connection error, reconnecting...')
    }

    return () => es.close()
  }, [url])
}
