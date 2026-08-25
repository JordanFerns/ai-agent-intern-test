"""Schemas for Agent Tools and Data Access."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class OrderLookupInput(BaseModel):
    """Input parameters for order status lookup."""
    order_id: str = Field(description="Order identifier, e.g. 'ORD-1007'")


class SanitizedOrderItem(BaseModel):
    """Customer-safe item information."""
    name: str
    quantity: int
    final_sale: bool


class SanitizedOrderResult(BaseModel):
    """
    Sanitized, customer-safe order lookup result.
    Guarantees internal fields (notes, risk score, email, address) are strictly excluded.
    """
    found: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    status_updated_at: Optional[str] = None
    placed_at: Optional[str] = None
    membership_tier: Optional[str] = None
    items: List[SanitizedOrderItem] = Field(default_factory=list)
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    requires_human_handoff: bool = False
    is_cancellable_window: bool = False
    error_message: Optional[str] = None

    def to_customer_summary(self) -> str:
        """Render a clean, sanitized text summary for the LLM context."""
        if not self.found:
            return (
                f"Order Lookup Result: NOT FOUND.\n"
                f"Details: {self.error_message or 'The order was not found in our records.'}\n"
                f"Action Required: Ask the customer to check the order ID or recommend human support."
            )

        items_str = ", ".join([f"{it.name} (Qty: {it.quantity}{', Final Sale' if it.final_sale else ''})" for it in self.items])
        
        lines = [
            f"Order ID: {self.order_id}",
            f"Status: {self.status.upper() if self.status else 'UNKNOWN'}",
            f"Membership Tier: {self.membership_tier}",
            f"Items: {items_str}",
            f"Placed At: {self.placed_at}",
        ]

        if self.status in ("cancelled", "returned"):
            lines.append(f"Notice: The order is {self.status}. It will not be delivered.")
        else:
            if self.carrier:
                lines.append(f"Carrier: {self.carrier}")
            if self.tracking_number:
                lines.append(f"Tracking Number: {self.tracking_number}")
            if self.estimated_delivery:
                lines.append(f"Estimated Delivery: {self.estimated_delivery}")
            elif self.status == "shipped":
                lines.append("Estimated Delivery: Unavailable (No estimated date provided by carrier)")

        if self.customer_safe_message:
            lines.append(f"Message: {self.customer_safe_message}")

        if self.requires_human_handoff:
            lines.append("Operational Flag: Exception status requires human specialist review.")

        return "\n".join(lines)
