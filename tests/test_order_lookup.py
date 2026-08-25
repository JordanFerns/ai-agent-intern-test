"""Unit tests for Order Lookup Tool, Privacy Sanitization, and Precedence Rules."""
import pytest
from src.tools.order_lookup import lookup_order_tool, OrderLookupService


def test_valid_order_lookup():
    """Verify ORD-1007 returns correct sanitized status, carrier, and ETA."""
    result = lookup_order_tool("ORD-1007")
    assert result.found is True
    assert result.order_id == "ORD-1007"
    assert result.status == "shipped"
    assert result.carrier == "UPS"
    assert result.estimated_delivery == "2026-08-22"
    assert len(result.items) > 0


def test_privacy_forbidden_fields_never_exposed():
    """Verify email, address, risk_score, and internal notes are NEVER present."""
    result = lookup_order_tool("ORD-1007")
    
    # Check model dump dictionary
    data_dict = result.model_dump()
    assert "email" not in data_dict
    assert "address" not in data_dict
    assert "shipping_address" not in data_dict
    assert "risk_score" not in data_dict
    assert "warehouse_note" not in data_dict
    assert "internal" not in data_dict
    assert "customer" not in data_dict

    # Check generated summary string
    summary = result.to_customer_summary()
    assert "ava.morgan@example.test" not in summary
    assert "220 King Street" not in summary
    assert "82" not in summary
    assert "fraud review" not in summary


def test_order_id_normalization():
    """Verify casing and surrounding whitespace are normalized."""
    res1 = lookup_order_tool("ord-1007")
    assert res1.found is True
    assert res1.order_id == "ORD-1007"

    res2 = lookup_order_tool("   ORD-1007   ")
    assert res2.found is True
    assert res2.order_id == "ORD-1007"

    res3 = lookup_order_tool("ord 1007")
    assert res3.found is True
    assert res3.order_id == "ORD-1007"


def test_cancelled_order_stale_eta_suppressed():
    """Verify cancelled order ORD-1004 does NOT expose stale estimated delivery."""
    result = lookup_order_tool("ORD-1004")
    assert result.found is True
    assert result.status == "cancelled"
    assert result.estimated_delivery is None
    assert result.carrier is None
    assert result.tracking_number is None
    
    summary = result.to_customer_summary()
    assert "2026-08-16" not in summary
    assert "cancelled" in summary.lower()
    assert "will not be delivered" in summary.lower()


def test_shipped_order_without_eta():
    """Verify shipped order ORD-1011 preserves Canada Post but reports ETA unavailable."""
    result = lookup_order_tool("ORD-1011")
    assert result.found is True
    assert result.status == "shipped"
    assert result.carrier == "Canada Post"
    assert result.estimated_delivery is None

    summary = result.to_customer_summary()
    assert "Canada Post" in summary
    assert "Unavailable" in summary


def test_unknown_order():
    """Verify unknown order ORD-9999 returns not found and requires human handoff."""
    result = lookup_order_tool("ORD-9999")
    assert result.found is False
    assert result.order_id == "ORD-9999"
    assert result.requires_human_handoff is True
    assert "not found" in result.error_message.lower()


def test_missing_or_empty_order_id():
    """Verify missing or blank order ID prompts for valid ID."""
    result = lookup_order_tool("")
    assert result.found is False
    assert "valid order id" in result.error_message.lower()
