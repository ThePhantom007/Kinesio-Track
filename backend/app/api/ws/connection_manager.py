"""
Manages active WebSocket connections across the process.

Single-process mode (development / single Uvicorn worker)
----------------------------------------------------------
Active connections are stored in a plain dict keyed by session_id.
Broadcast sends directly to each WebSocket object.

Multi-process mode (production with multiple Uvicorn workers)
-------------------------------------------------------------
Each worker process holds only the connections opened against it.
Cross-worker broadcast (e.g. clinician monitoring a session that connected
to a different worker) uses Redis Pub/Sub:
  - Worker A (patient connected) subscribes to channel "ws:session:{id}"
  - Worker B (clinician connected) publishes to "ws:session:{id}"
  - Worker A receives the message and delivers to the patient's WebSocket

The connection_manager subscribes on connect and unsubscribes on disconnect.
A background asyncio task per connection relays Redis messages to the socket.

Clinician monitoring
--------------------
A clinician can connect to WS /ws/session/{session_id}?monitor=true.
Their connection is stored in a separate dict; they receive all server→patient
messages but their landmark frames are ignored.
"""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

from fastapi import WebSocket
from redis.asyncio import Redis

from app.core.logging import get_logger

log = get_logger(__name__)


class ConnectionManager:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        # session_id → patient WebSocket
        self._patient_connections: dict[str, WebSocket] = {}
        # session_id → list of clinician/monitor WebSockets
        self._monitor_connections: dict[str, list[WebSocket]] = {}
        # session_id → asyncio background task (Redis listener)
        self._listener_tasks: dict[str, asyncio.Task] = {}

    # ── Connect / disconnect ───────────────────────────────────────────────────

    async def connect(
        self,
        session_id: UUID,
        websocket: WebSocket,
        is_monitor: bool = False,
    ) -> None:
        """
        Register a new WebSocket connection for the given session.

        Args:
            session_id:  Session UUID.
            websocket:   Accepted WebSocket object.
            is_monitor:  True for clinician monitor connections.
        """
        sid = str(session_id)
        await websocket.accept()

        if is_monitor:
            self._monitor_connections.setdefault(sid, []).append(websocket)
            log.info("ws_monitor_connected", session_id=sid)
        else:
            self._patient_connections[sid] = websocket
            # Subscribe to Redis channel so other workers can broadcast to us
            task = asyncio.create_task(
                self._redis_listener(sid),
                name=f"ws_listener_{sid}",
            )
            self._listener_tasks[sid] = task
            log.info("ws_patient_connected", session_id=sid)

    async def disconnect(
        self,
        session_id: UUID,
        websocket: WebSocket,
        is_monitor: bool = False,
    ) -> None:
        """
        Remove a WebSocket connection and clean up associated resources.
        """
        sid = str(session_id)

        if is_monitor:
            monitors = self._monitor_connections.get(sid, [])
            if websocket in monitors:
                monitors.remove(websocket)
            if not monitors:
                self._monitor_connections.pop(sid, None)
            log.info("ws_monitor_disconnected", session_id=sid)
        else:
            self._patient_connections.pop(sid, None)
            task = self._listener_tasks.pop(sid, None)
            if task and not task.done():
                task.cancel()
            log.info("ws_patient_disconnected", session_id=sid)

    # ── Send to patient ────────────────────────────────────────────────────────

    async def send_to_patient(self, session_id: UUID, message: dict) -> None:
        """
        Send a message to the patient connected on this session.

        If the connection is on this worker, sends directly.
        Also publishes to Redis so that monitors on other workers receive it.
        """
        sid = str(session_id)
        payload = json.dumps(message)

        # Direct send (same worker)
        websocket = self._patient_connections.get(sid)
        if websocket:
            try:
                await websocket.send_text(payload)
            except Exception as exc:
                log.warning("ws_send_to_patient_failed", session_id=sid, error=str(exc))

        # Pub/sub broadcast (cross-worker monitors + redundancy)
        await self._redis.publish(f"ws:session:{sid}", payload)

    async def broadcast_to_monitors(self, session_id: UUID, message: dict) -> None:
        """
        Send a message to all clinician monitors of this session on this worker.
        """
        sid = str(session_id)
        payload = json.dumps(message)
        monitors = list(self._monitor_connections.get(sid, []))
        dead: list[WebSocket] = []

        for ws in monitors:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            monitors.remove(ws)

    # ── Connection state ───────────────────────────────────────────────────────

    def is_connected(self, session_id: UUID) -> bool:
        """Return True if a patient WebSocket is active for this session."""
        return str(session_id) in self._patient_connections

    def active_session_count(self) -> int:
        """Return the number of active patient connections on this worker."""
        return len(self._patient_connections)

    # ── Redis listener ─────────────────────────────────────────────────────────

    async def _redis_listener(self, sid: str) -> None:
        """
        Background task that subscribes to Redis channel "ws:session:{sid}"
        and forwards published messages to monitor connections on this worker.

        Patient messages are NOT relayed back through Redis to avoid loops —
        the direct send in send_to_patient() handles the patient's socket.
        """
        pubsub = self._redis.pubsub()
        channel = f"ws:session:{sid}"
        await pubsub.subscribe(channel)
        log.debug("redis_pubsub_subscribed", channel=channel)

        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()

                # Forward to monitors on this worker
                monitors = list(self._monitor_connections.get(sid, []))
                for ws in monitors:
                    try:
                        await ws.send_text(data)
                    except Exception:
                        pass

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("redis_listener_error", session_id=sid, error=str(exc))
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            log.debug("redis_pubsub_unsubscribed", channel=channel)