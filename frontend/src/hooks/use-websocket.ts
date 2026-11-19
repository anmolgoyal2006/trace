/**
 * useWebSocket.ts — Hook for subscribing to WebSocket events.
 */

import { useEffect, useState } from "react";
import { wsClient } from "@/services/websocket";
import type { WsEvent } from "@/types/api";

export function useWebSocketConnection(): boolean {
  const [connected, setConnected] = useState(wsClient.isConnected);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.onStatusChange(setConnected);
    return () => {
      unsub();
    };
  }, []);

  return connected;
}

export function useWsEvent(
  handler: (event: WsEvent) => void,
  deps: unknown[] = [],
): void {
  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.onMessage(handler);
    return unsub;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export { wsClient };
