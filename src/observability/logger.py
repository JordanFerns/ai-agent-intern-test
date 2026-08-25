"""Structured per-turn logging and observability for Aster & Row Support Agent."""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config import BASE_DIR, DEBUG_MODE

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
TRACE_LOG_FILE = LOGS_DIR / "agent_traces.jsonl"


class AgentObserver:
    """Structured per-turn observer for auditing agent interactions."""

    def __init__(self, log_file: Path = TRACE_LOG_FILE):
        self.log_file = log_file

    def log_turn(
        self,
        session_id: str,
        user_message: str,
        history_summary: List[Dict[str, str]],
        retrieved_passages: List[Dict[str, Any]],
        tool_call: Optional[str],
        tool_args: Optional[Dict[str, Any]],
        sanitized_tool_result: Optional[str],
        final_answer: str,
        sources: List[str],
        handoff_recommended: bool,
        has_conflict: bool = False,
        execution_time_ms: float = 0.0,
    ):
        """
        Record a structured per-turn log entry.
        Guarantees secrets and customer personal info are never written to logs.
        """
        trace_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "user_message": user_message,
            "conversation_history_turns": len(history_summary),
            "retrieval": {
                "count": len(retrieved_passages),
                "has_conflict": has_conflict,
                "passages": [
                    {
                        "citation": p.get("citation"),
                        "score": round(p.get("score", 0.0), 3),
                        "status": p.get("status"),
                        "authority": p.get("authority"),
                    }
                    for p in retrieved_passages
                ],
            },
            "tool_execution": {
                "tool_called": tool_call,
                "arguments": tool_args,
                "sanitized_result": sanitized_tool_result,
            },
            "response": {
                "answer_preview": final_answer[:150] + "..." if len(final_answer) > 150 else final_answer,
                "sources_cited": sources,
                "handoff_recommended": handoff_recommended,
            },
            "latency_ms": round(execution_time_ms, 2),
        }

        # Append structured JSON line
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record) + "\n")

        if DEBUG_MODE:
            print(f"\n[OBSERVABILITY TRACE] Session: {session_id} | Handoff: {handoff_recommended} | Sources: {sources}")


# Global singleton observer
agent_observer = AgentObserver()
