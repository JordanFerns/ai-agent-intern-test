"""Session management for multi-turn conversation context."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid
import time


class Message(BaseModel):
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationSession(BaseModel):
    session_id: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: List[Message] = Field(default_factory=list)
    active_order_id: Optional[str] = None
    active_topic: Optional[str] = None

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = time.time()

    def get_recent_history(self, max_turns: int = 6) -> List[Message]:
        """Return the most recent N messages to avoid context explosion."""
        return self.messages[-max_turns:] if self.messages else []


class SessionManager:
    """In-memory session registry ensuring strict session isolation."""

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> ConversationSession:
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id=session_id)
        
        return self._sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]


# Global session manager instance
session_manager = SessionManager()
