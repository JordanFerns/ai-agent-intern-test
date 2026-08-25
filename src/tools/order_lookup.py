"""Order lookup tool with strict privacy sanitization and status precedence enforcement."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config import ORDERS_FILE
from src.tools.schemas import SanitizedOrderResult, SanitizedOrderItem


class OrderLookupService:
    """Service to safely look up and sanitize order details."""

    def __init__(self, orders_file: Path = ORDERS_FILE):
        self.orders_file = orders_file
        self.snapshot_time: Optional[datetime] = None
        self._orders_by_id: Dict[str, Dict[str, Any]] = {}
        self._load_dataset()

    def _load_dataset(self):
        if not self.orders_file.exists():
            return
        
        with open(self.orders_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            snapshot_str = data.get("snapshot_at", "2026-08-15T12:00:00Z")
            try:
                self.snapshot_time = datetime.fromisoformat(snapshot_str.replace("Z", "+00:00"))
            except Exception:
                self.snapshot_time = datetime.now(timezone.utc)

            for order in data.get("orders", []):
                oid = order.get("order_id", "").strip().upper()
                if oid:
                    self._orders_by_id[oid] = order

    @staticmethod
    def normalize_order_id(raw_id: str) -> Optional[str]:
        """
        Normalize harmless whitespace, punctuation, and casing.
        Extracts 'ORD-XXXX' format safely without hallucinating digits or matching arbitrary prose.
        """
        if not raw_id or not isinstance(raw_id, str):
            return None
        
        cleaned = raw_id.strip().upper()
        # Look for explicit pattern ORD-1001, ord 1001, ORD1001, #ORD-1001
        match = re.search(r"\bORD[- ]?(\d{3,6})\b", cleaned, re.IGNORECASE)
        if match:
            digits = match.group(1)
            return f"ORD-{digits}"
        
        # Exact match if passed directly as ORD-XXXX
        if re.match(r"^ORD-\d+$", cleaned, re.IGNORECASE):
            return cleaned.upper()

        return None

    def lookup(self, raw_order_id: str) -> SanitizedOrderResult:
        """
        Look up an order by ID and return strictly sanitized customer-safe fields.
        Guarantees internal fields, customer personal identity, and stale estimates
        for cancelled/returned orders are never exposed.
        """
        normalized_id = self.normalize_order_id(raw_order_id)
        
        if not normalized_id:
            return SanitizedOrderResult(
                found=False,
                error_message="Please provide a valid order ID (for example, 'ORD-1007').",
                requires_human_handoff=False
            )

        raw_order = self._orders_by_id.get(normalized_id)
        if not raw_order:
            return SanitizedOrderResult(
                found=False,
                order_id=normalized_id,
                error_message=f"Order '{normalized_id}' was not found in our system. Please check the order ID or contact support.",
                requires_human_handoff=True
            )

        status = str(raw_order.get("status", "unknown")).lower()
        placed_at_str = raw_order.get("placed_at")
        
        # Check cancellation window (30 minutes from placement if pending)
        is_cancellable = False
        if status == "pending" and placed_at_str and self.snapshot_time:
            try:
                placed_dt = datetime.fromisoformat(placed_at_str.replace("Z", "+00:00"))
                delta_minutes = (self.snapshot_time - placed_dt).total_seconds() / 60.0
                if 0 <= delta_minutes <= 30:
                    is_cancellable = True
            except Exception:
                pass

        # Build sanitized items
        sanitized_items: List[SanitizedOrderItem] = []
        for item in raw_order.get("items", []):
            sanitized_items.append(
                SanitizedOrderItem(
                    name=str(item.get("name", "Item")),
                    quantity=int(item.get("quantity", 1)),
                    final_sale=bool(item.get("final_sale", False))
                )
            )

        # Apply Status Precedence Rules:
        carrier = raw_order.get("carrier")
        tracking_number = raw_order.get("tracking_number")
        estimated_delivery = raw_order.get("estimated_delivery")
        customer_safe_msg = raw_order.get("customer_safe_message")

        requires_handoff = False
        if status in ("cancelled", "returned"):
            carrier = None
            tracking_number = None
            estimated_delivery = None
            if status == "cancelled":
                customer_safe_msg = "This order was cancelled and will not be shipped."
            else:
                customer_safe_msg = "This order was returned."
        elif status == "exception":
            requires_handoff = True
            customer_safe_msg = "This order is currently flagged for specialist review."

        return SanitizedOrderResult(
            found=True,
            order_id=normalized_id,
            status=status,
            status_updated_at=raw_order.get("status_updated_at"),
            placed_at=placed_at_str,
            membership_tier=raw_order.get("membership_tier", "standard"),
            items=sanitized_items,
            carrier=carrier,
            tracking_number=tracking_number,
            estimated_delivery=estimated_delivery,
            customer_safe_message=customer_safe_msg,
            requires_human_handoff=requires_handoff,
            is_cancellable_window=is_cancellable
        )


# Global singleton instance
_order_service = OrderLookupService()


def lookup_order_tool(order_id: str) -> SanitizedOrderResult:
    """Tool function to be called by agent for looking up order status."""
    return _order_service.lookup(order_id)
