---
name: websocket-patterns
description: Load when user needs WebSocket or real-time communication patterns including SSE, Socket.IO, and pub/sub
trigger_keywords: [websocket, ws, socket.io, real-time, sse, server sent events, pub sub, live update, push notification, bidirectional]
---

# WebSocket & Real-Time Patterns Skill

Production patterns for real-time communication including WebSockets, Server-Sent Events (SSE), and pub/sub architectures.

## Overview

Real-time communication enables instant data delivery without polling. Choose the right transport based on your requirements.

**When to Use**:
- Live dashboards and monitoring
- Chat and collaboration features
- Real-time notifications
- Live data feeds (stock prices, sports scores)
- Multiplayer or collaborative editing

## Transport Selection

| Feature | WebSocket | SSE | Long Polling |
|---------|-----------|-----|--------------|
| Direction | Bidirectional | Server → Client | Server → Client |
| Protocol | ws:// | HTTP | HTTP |
| Auto-reconnect | Manual | Built-in | Manual |
| Binary data | Yes | No (text only) | Yes |
| Scalability | Harder | Easier | Easiest |
| Browser support | All modern | All modern | Universal |

**Decision Rule**:
- Need bidirectional? → WebSocket
- Server-only push, simple? → SSE
- Proxy/firewall issues? → SSE (regular HTTP)

## WebSocket Server (Node.js)

```typescript
import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';

const server = createServer();
const wss = new WebSocketServer({ server });

// Room-based pub/sub
const rooms = new Map<string, Set<WebSocket>>();

wss.on('connection', (ws, req) => {
  const userId = authenticateFromHeaders(req.headers);
  if (!userId) { ws.close(4001, 'Unauthorized'); return; }

  // Heartbeat — detect dead connections
  let isAlive = true;
  ws.on('pong', () => { isAlive = true; });

  const heartbeat = setInterval(() => {
    if (!isAlive) { ws.terminate(); return; }
    isAlive = false;
    ws.ping();
  }, 30_000);

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      switch (msg.type) {
        case 'join':
          joinRoom(msg.room, ws);
          break;
        case 'leave':
          leaveRoom(msg.room, ws);
          break;
        case 'message':
          broadcast(msg.room, {
            type: 'message',
            from: userId,
            content: msg.content,
            timestamp: Date.now(),
          }, ws);
          break;
      }
    } catch {
      ws.send(JSON.stringify({ type: 'error', message: 'Invalid message format' }));
    }
  });

  ws.on('close', () => {
    clearInterval(heartbeat);
    removeFromAllRooms(ws);
  });
});

function broadcast(room: string, data: unknown, exclude?: WebSocket) {
  const clients = rooms.get(room);
  if (!clients) return;
  const payload = JSON.stringify(data);
  for (const client of clients) {
    if (client !== exclude && client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

server.listen(3001);
```

## Server-Sent Events (SSE)

```typescript
// Express SSE endpoint
app.get('/api/events/:channel', (req, res) => {
  const { channel } = req.params;

  // SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no', // Disable Nginx buffering
  });

  // Send initial connection event
  res.write(`event: connected\ndata: ${JSON.stringify({ channel })}\n\n`);

  // Subscribe to channel
  const handler = (data: unknown) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };
  pubsub.subscribe(channel, handler);

  // Keep-alive ping every 15s
  const keepAlive = setInterval(() => {
    res.write(': keepalive\n\n');
  }, 15_000);

  // Cleanup on disconnect
  req.on('close', () => {
    clearInterval(keepAlive);
    pubsub.unsubscribe(channel, handler);
  });
});

// Client-side
const source = new EventSource('/api/events/notifications');
source.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
source.onerror = () => {
  // EventSource auto-reconnects
  console.log('Connection lost, reconnecting...');
};
```

## React Client Hook

```tsx
function useWebSocket(url: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed'>('connecting');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let reconnectTimer: NodeJS.Timeout;
    let attempts = 0;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('open');
        attempts = 0;
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setMessages((prev) => [...prev, msg]);
      };

      ws.onclose = () => {
        setStatus('closed');
        // Exponential backoff reconnect
        const delay = Math.min(1000 * 2 ** attempts, 30000);
        reconnectTimer = setTimeout(() => { attempts++; connect(); }, delay);
      };
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [url]);

  const send = useCallback((data: unknown) => {
    wsRef.current?.send(JSON.stringify(data));
  }, []);

  return { messages, status, send };
}
```

## Scaling WebSockets

```
Single Server: Direct in-memory pub/sub (Map<room, Set<ws>>)
Multi-Server:  Redis pub/sub as message bus between instances

Client ──→ Server A ──→ Redis Pub/Sub ──→ Server B ──→ Client
                                     └──→ Server C ──→ Client

Pattern: Each server subscribes to Redis channels matching rooms.
When a message arrives on any server, it publishes to Redis.
All servers receive and broadcast to their local clients.
```

## Best Practices

1. **Heartbeat/ping-pong** — detect dead connections (30s interval)
2. **Exponential backoff** — for client reconnection (cap at 30s)
3. **Message validation** — parse and validate all incoming messages
4. **Authentication** — verify on connection, not per-message
5. **Rate limiting** — prevent message flooding per connection
6. **Graceful shutdown** — close connections with proper close codes

---

**Skill Type**: Backend — Real-Time
**Complexity**: Moderate
**Typical Usage**: Real-time features, WebSocket server setup, SSE implementation
