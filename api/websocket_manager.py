from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
from fastapi import WebSocket
import json
from datetime import datetime

class MessageType(Enum):
    """WebSocket消息类型枚举"""
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"
    LOG = "log"
    HEARTBEAT = "heartbeat"

@dataclass
class WSMessage:
    """WebSocket消息结构"""
    type: str # Use string for direct JSON serialization
    data: dict
    timestamp: str

def get_iso_timestamp():
    return datetime.utcnow().isoformat() + "Z"

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        self.active_connections[user_id] = {
            "websocket": websocket,
            "created_at": get_iso_timestamp()
        }

    async def disconnect(self, user_id: str) -> None:
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def broadcast(self, message: WSMessage, user_id: Optional[str] = None) -> None:
        message_dict = {
            "type": message.type,
            "data": message.data,
            "timestamp": message.timestamp
        }
        
        targets = [user_id] if user_id and user_id in self.active_connections else list(self.active_connections.keys())
        
        for target_id in targets:
            if target_id in self.active_connections:
                ws = self.active_connections[target_id]["websocket"]
                try:
                    await ws.send_json(message_dict)
                except Exception:
                    await self.disconnect(target_id)

    async def send_progress(self, user_id: str, progress_pct: float, current_stage: str, current_professor: str) -> None:
        msg = WSMessage(
            type=MessageType.PROGRESS.value,
            data={
                "progress_pct": progress_pct,
                "current_stage": current_stage,
                "current_professor": current_professor
            },
            timestamp=get_iso_timestamp()
        )
        await self.broadcast(msg, user_id)

    async def send_error(self, user_id: str, error_msg: str, current_stage: Optional[str] = None) -> None:
        msg = WSMessage(
            type=MessageType.ERROR.value,
            data={
                "error_msg": error_msg,
                "current_stage": current_stage,
                "current_professor": None
            },
            timestamp=get_iso_timestamp()
        )
        await self.broadcast(msg, user_id)

    async def send_log(self, user_id: str, log_msg: str) -> None:
        msg = WSMessage(
            type=MessageType.LOG.value,
            data={"log_msg": log_msg},
            timestamp=get_iso_timestamp()
        )
        await self.broadcast(msg, user_id)

    async def send_completion(self, user_id: str, markdown_content: str, summary_stats: dict) -> None:
        msg = WSMessage(
            type=MessageType.COMPLETED.value,
            data={
                "markdown_content": markdown_content,
                "summary_stats": summary_stats
            },
            timestamp=get_iso_timestamp()
        )
        await self.broadcast(msg, user_id)
