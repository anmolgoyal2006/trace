"""
websocket_manager.py — WebSocketManager
Manages all active WebSocket connections and broadcasts JSON events.

Events pushed by the pipeline:
  query_progress   — % complete + message
  sighting_found   — per-camera match confirmed
  route_complete   — full route + all sightings
  alert            — watchlist hit
  error            — session failure
"""

import json
from typing import Any

from fastapi import WebSocket
from loguru import logger


class WebSocketManager:
    """
    Thread-safe (asyncio) WebSocket connection registry.

    Usage:
        manager = WebSocketManager()

        # In WebSocket endpoint:
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()   # keep alive
        except:
            manager.disconnect(websocket)

        # From pipeline / services:
        await manager.broadcast({"type": "query_progress", ...})
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(f"[WS] Client connected. Total: {len(self._connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info(f"[WS] Client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send a JSON-serialisable dict to all connected clients."""
        if not self._connections:
            return
        payload = json.dumps(data, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Module-level singleton
ws_manager = WebSocketManager()
