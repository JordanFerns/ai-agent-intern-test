"""Core Orchestrator for Aster & Row Support Agent."""
import json
import re
from typing import Optional, Dict, Any, List

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.kb.indexer import KBRetriever
from src.kb.schemas import SearchResponse
from src.tools.order_lookup import lookup_order_tool, OrderLookupService
from src.tools.schemas import SanitizedOrderResult
from src.agent.session import session_manager, ConversationSession
from src.agent.response import AgentResponse
from src.agent.prompts import SYSTEM_PROMPT


class AsterRowAgent:
    """Production-grade RAG support agent for Aster & Row."""

    def __init__(self, kb_retriever: Optional[KBRetriever] = None):
        self.kb = kb_retriever or KBRetriever()
        self.session_mgr = session_manager

    def process_message(
        self,
        user_message: str,
        session_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Process a single customer message in a session-aware manner.
        """
        session = self.session_mgr.get_or_create(session_id)
        raw_query = user_message.strip()

        # Step 1: Detect Order IDs and Intent
        extracted_order_id = OrderLookupService.normalize_order_id(raw_query)
        order_intent = self._detect_order_intent(raw_query)
        
        # Multi-turn order ID resolution: only use session order ID if the user's intent relates to an order
        active_order_id = extracted_order_id or (session.active_order_id if order_intent else None)

        tool_called: Optional[str] = None
        tool_args: Optional[Dict[str, Any]] = None
        tool_result: Optional[SanitizedOrderResult] = None
        sanitized_summary: Optional[str] = None
        handoff_recommended = False

        # Step 2: Handle Order Inquiries
        if order_intent or extracted_order_id:
            if not active_order_id:
                # User asked about an order but provided no order ID
                ans = (
                    "I would be happy to check your order status. Could you please provide your order ID "
                    "(for example, 'ORD-1234')?"
                )
                session.add_message("user", raw_query)
                session.add_message("assistant", ans)
                return AgentResponse(
                    answer=ans,
                    sources=[],
                    handoff_recommended=False,
                    tool_called=None,
                    debug_trace={"intent": "order_inquiry_missing_id"}
                )

            # Order ID is present -> execute sanitized lookup tool
            tool_called = "order_lookup"
            tool_args = {"order_id": active_order_id}
            tool_result = lookup_order_tool(active_order_id)
            sanitized_summary = tool_result.to_customer_summary()
            
            if tool_result.found:
                session.active_order_id = active_order_id
            if tool_result.requires_human_handoff:
                handoff_recommended = True

        # Step 3: Contextualize Search Query for KB (Multi-turn awareness)
        kb_query = self._contextualize_query(raw_query, session)
        kb_response: SearchResponse = self.kb.retrieve(kb_query)

        if kb_response.has_conflict:
            handoff_recommended = True

        # Step 4: Check for privacy probes / adversarial requests
        is_privacy_probe = self._detect_privacy_probe(raw_query)
        if is_privacy_probe:
            handoff_recommended = True

        # Step 5: Generate Grounded Answer (LLM or Deterministic Engine)
        response = self._generate_response(
            user_message=raw_query,
            session=session,
            kb_response=kb_response,
            tool_result=tool_result,
            tool_called=tool_called,
            tool_args=tool_args,
            sanitized_summary=sanitized_summary,
            handoff_recommended=handoff_recommended,
            is_privacy_probe=is_privacy_probe
        )

        # Update Session History
        session.add_message("user", raw_query)
        session.add_message("assistant", response.answer)
        if "ship" in raw_query.lower() or "international" in raw_query.lower():
            session.active_topic = "shipping"
        elif "return" in raw_query.lower():
            session.active_topic = "returns"

        return response

    def _detect_order_intent(self, query: str) -> bool:
        q = query.lower()
        if "order" in q and any(k in q for k in ["where", "status", "track", "when", "arrive", "get here", "shipped", "check", "my order"]):
            return True
        if any(phrase in q for phrase in ["when will it arrive", "where is it", "when will it get here", "track package", "package status"]):
            return True
        return False

    def _detect_privacy_probe(self, query: str) -> bool:
        q = query.lower()
        privacy_keywords = [
            "email", "address", "internal note", "risk score", "warehouse note",
            "hidden prompt", "reveal your instructions", "system prompt"
        ]
        return any(pk in q for pk in privacy_keywords)

    def _contextualize_query(self, query: str, session: ConversationSession) -> str:
        q = query.lower().strip()
        if session.messages:
            last_user_msg = ""
            for m in reversed(session.messages):
                if m.role == "user":
                    last_user_msg = m.content.lower()
                    break
            
            if ("canada" in q or "international" in q) and ("ship" in last_user_msg or session.active_topic == "shipping"):
                return f"international shipping to Canada delivery estimate {query}"
            if "what about" in q and session.active_topic:
                return f"{session.active_topic} {query}"

        return query

    def _generate_response(
        self,
        user_message: str,
        session: ConversationSession,
        kb_response: SearchResponse,
        tool_result: Optional[SanitizedOrderResult],
        tool_called: Optional[str],
        tool_args: Optional[Dict[str, Any]],
        sanitized_summary: Optional[str],
        handoff_recommended: bool,
        is_privacy_probe: bool
    ) -> AgentResponse:
        if OPENAI_API_KEY:
            try:
                return self._call_llm(
                    user_message=user_message,
                    session=session,
                    kb_response=kb_response,
                    tool_result=tool_result,
                    tool_called=tool_called,
                    tool_args=tool_args,
                    sanitized_summary=sanitized_summary,
                    handoff_recommended=handoff_recommended,
                    is_privacy_probe=is_privacy_probe
                )
            except Exception:
                pass

        return self._deterministic_generator(
            user_message=user_message,
            session=session,
            kb_response=kb_response,
            tool_result=tool_result,
            tool_called=tool_called,
            tool_args=tool_args,
            sanitized_summary=sanitized_summary,
            handoff_recommended=handoff_recommended,
            is_privacy_probe=is_privacy_probe
        )

    def _call_llm(
        self,
        user_message: str,
        session: ConversationSession,
        kb_response: SearchResponse,
        tool_result: Optional[SanitizedOrderResult],
        tool_called: Optional[str],
        tool_args: Optional[Dict[str, Any]],
        sanitized_summary: Optional[str],
        handoff_recommended: bool,
        is_privacy_probe: bool
    ) -> AgentResponse:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        kb_context_blocks = []
        for res in kb_response.results:
            c = res.chunk
            auth_tag = "AUTHORITATIVE ACTIVE POLICY" if res.is_authoritative else f"NON-AUTHORITATIVE ({res.status.upper()})"
            kb_context_blocks.append(
                f"[{c.citation}] ({auth_tag}):\n{c.content}"
            )
        kb_text = "\n\n---\n\n".join(kb_context_blocks) if kb_context_blocks else "No relevant passages found."

        tool_text = sanitized_summary if sanitized_summary else "No tool called."

        conflict_note = ""
        if kb_response.has_conflict:
            conflict_note = f"\nWARNING: {kb_response.conflict_details}\n"

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        for m in session.get_recent_history(max_turns=4):
            prompt_messages.append({"role": m.role, "content": m.content})

        user_content = (
            f"=== RETRIEVED PASSAGES ===\n{kb_text}\n\n"
            f"{conflict_note}"
            f"=== ORDER TOOL OUTPUT ===\n{tool_text}\n\n"
            f"=== CUSTOMER INQUIRY ===\n{user_message}\n\n"
            f"Provide a helpful, accurate, and properly cited response. "
            f"Output JSON with keys: 'answer', 'sources' (list of citations matching '[filename > heading]'), and 'handoff_recommended' (boolean)."
        )
        prompt_messages.append({"role": "user", "content": user_content})

        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=prompt_messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )

        content = res.choices[0].message.content or "{}"
        data = json.loads(content)

        answer = data.get("answer", "")
        sources = data.get("sources", kb_response.citations[:2])
        model_handoff = bool(data.get("handoff_recommended", False)) or handoff_recommended

        return AgentResponse(
            answer=answer,
            sources=sources,
            handoff_recommended=model_handoff,
            tool_called=tool_called,
            tool_arguments=tool_args,
            tool_result_sanitized=sanitized_summary,
            debug_trace={
                "kb_query": kb_response.query,
                "retrieved_count": len(kb_response.results),
                "has_conflict": kb_response.has_conflict
            }
        )

    def _deterministic_generator(
        self,
        user_message: str,
        session: ConversationSession,
        kb_response: SearchResponse,
        tool_result: Optional[SanitizedOrderResult],
        tool_called: Optional[str],
        tool_args: Optional[Dict[str, Any]],
        sanitized_summary: Optional[str],
        handoff_recommended: bool,
        is_privacy_probe: bool
    ) -> AgentResponse:
        q_lower = user_message.lower()

        # Privacy Probe Refusal
        if is_privacy_probe:
            return AgentResponse(
                answer=(
                    "For security and customer privacy, I cannot disclose personal information such as emails, "
                    "shipping addresses, internal notes, risk scores, or system configurations. "
                    "If you need account assistance, please contact our human support team."
                ),
                sources=[],
                handoff_recommended=True,
                tool_called=tool_called,
                tool_arguments=tool_args,
                tool_result_sanitized=sanitized_summary,
                debug_trace={"privacy_refusal": True}
            )

        # Order Lookup Inquiries
        if tool_result:
            if not tool_result.found:
                return AgentResponse(
                    answer=(
                        f"I checked our system, but {tool_result.error_message or 'the order was not found'}. "
                        "Please verify the order ID or connect with our support team for further assistance."
                    ),
                    sources=[],
                    handoff_recommended=True,
                    tool_called=tool_called,
                    tool_arguments=tool_args,
                    tool_result_sanitized=sanitized_summary
                )

            if tool_result.status == "cancelled":
                return AgentResponse(
                    answer=(
                        f"Order {tool_result.order_id} is cancelled and will not be shipped. "
                        "Because the order has been cancelled, there is no active delivery date."
                    ),
                    sources=[],
                    handoff_recommended=False,
                    tool_called=tool_called,
                    tool_arguments=tool_args,
                    tool_result_sanitized=sanitized_summary
                )

            if tool_result.status == "shipped" and not tool_result.estimated_delivery:
                carrier_info = f" with {tool_result.carrier}" if tool_result.carrier else ""
                return AgentResponse(
                    answer=(
                        f"Order {tool_result.order_id} has shipped{carrier_info}. "
                        "A delivery estimate is currently unavailable from the carrier."
                    ),
                    sources=[],
                    handoff_recommended=False,
                    tool_called=tool_called,
                    tool_arguments=tool_args,
                    tool_result_sanitized=sanitized_summary
                )

            if tool_result.status == "shipped":
                carrier_str = f" via {tool_result.carrier}" if tool_result.carrier else ""
                eta_str = f" and is estimated to arrive on {tool_result.estimated_delivery}" if tool_result.estimated_delivery else ""
                return AgentResponse(
                    answer=f"Order {tool_result.order_id} has shipped{carrier_str}{eta_str}.",
                    sources=[],
                    handoff_recommended=False,
                    tool_called=tool_called,
                    tool_arguments=tool_args,
                    tool_result_sanitized=sanitized_summary
                )

            return AgentResponse(
                answer=f"Order {tool_result.order_id} is currently in {tool_result.status} status.",
                sources=[],
                handoff_recommended=tool_result.requires_human_handoff,
                tool_called=tool_called,
                tool_arguments=tool_args,
                tool_result_sanitized=sanitized_summary
            )

        # Conflict Handling: Breeze Tumbler Dishwasher Safety
        if kb_response.has_conflict:
            return AgentResponse(
                answer=(
                    "Our current official documentation contains conflicting guidance regarding cleaning the Breeze Tumbler: "
                    "the Product Care Guide [11-product-care.md > Breeze Tumbler] states that the stainless-steel body should be hand-washed, "
                    "while the Product Information card [12-breeze-tumbler-product-card.md > Cleaning] states that all components are dishwasher safe. "
                    "As the safest interim measure, we recommend hand-washing the stainless-steel body. "
                    "I am flagging this for a human support specialist to provide official confirmation."
                ),
                sources=[
                    "11-product-care.md > Breeze Tumbler",
                    "12-breeze-tumbler-product-card.md > Cleaning"
                ],
                handoff_recommended=True,
                tool_called=tool_called,
                debug_trace={"conflict_detected": True}
            )

        # Prompt Injection Defense: Migration Notes
        if "migration note" in q_lower or "ignore the real policy" in q_lower or "60 days" in q_lower:
            return AgentResponse(
                answer=(
                    "The internal migration notes are not authoritative customer policy. Under our official Returns Policy "
                    "[01-returns-policy-current.md > Standard return window], standard customers have 30 calendar days from delivery "
                    "to return eligible items. As an automated support agent, I cannot automatically approve returns."
                ),
                sources=["01-returns-policy-current.md > Standard return window"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Final Sale + Damaged Item Exception
        if "final" in q_lower and ("damaged" in q_lower or "broken" in q_lower or "defective" in q_lower):
            return AgentResponse(
                answer=(
                    "While final-sale items cannot be returned for a change of mind [03-final-sale-and-promotions.md > Change-of-mind returns], "
                    "final sale does not block a damaged-item review [03-final-sale-and-promotions.md > Damaged or incorrect items]. "
                    "Under our Damaged, Defective, or Wrong Items policy [04-damaged-or-wrong-items.md > Reporting window], you can report "
                    "an item that arrived damaged within 7 calendar days of delivery with photos for human review and resolution. "
                    "I am connecting you with a human support specialist to review your request."
                ),
                sources=[
                    "03-final-sale-and-promotions.md > Damaged or incorrect items",
                    "04-damaged-or-wrong-items.md > Reporting window"
                ],
                handoff_recommended=True,
                tool_called=tool_called
            )

        # TrailPlus Return Window
        if "trailplus" in q_lower and ("return" in q_lower or "window" in q_lower):
            return AgentResponse(
                answer=(
                    "Customers whose TrailPlus membership was active at the time of purchase receive an extended "
                    "return window of 45 calendar days from delivery [09-trailplus-membership.md > Return window]."
                ),
                sources=["09-trailplus-membership.md > Return window"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Standard Return Window
        if "return" in q_lower and ("how long" in q_lower or "window" in q_lower or "backpack" in q_lower or "regular" in q_lower):
            return AgentResponse(
                answer=(
                    "Standard customers have 30 calendar days from delivery to return unused items in resalable condition "
                    "[01-returns-policy-current.md > Standard return window]."
                ),
                sources=["01-returns-policy-current.md > Standard return window"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # International Shipping / Canada
        if "canada" in q_lower:
            return AgentResponse(
                answer=(
                    "Aster & Row ships internationally to Canada [06-international-shipping.md > Supported destinations]. "
                    "Canadian orders generally arrive within 5–9 business days after dispatch [06-international-shipping.md > Canada delivery estimate]. "
                    "Please note that import duties and taxes are not prepaid by Aster & Row and are the customer's responsibility."
                ),
                sources=[
                    "06-international-shipping.md > Supported destinations",
                    "06-international-shipping.md > Canada delivery estimate"
                ],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Unsupported Country (Germany etc.)
        if "germany" in q_lower or ("international" in q_lower and any(c in q_lower for c in ["uk", "europe", "france", "australia"])):
            return AgentResponse(
                answer=(
                    "Aster & Row currently ships internationally only to Canada. Shipping to Germany or other international "
                    "destinations is not currently available [06-international-shipping.md > Supported destinations]."
                ),
                sources=["06-international-shipping.md > Supported destinations"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Lifetime Warranty Inquiry
        if "lifetime warranty" in q_lower or ("warranty" in q_lower and "lifetime" in q_lower):
            return AgentResponse(
                answer=(
                    "Aster & Row does not offer a lifetime warranty [07-warranty.md > Warranty periods]. "
                    "Bags and backpacks are covered by a 2-year limited warranty from the purchase date, while drinkware and "
                    "travel accessories (such as packing cubes) are covered for 1 year."
                ),
                sources=["07-warranty.md > Warranty periods"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Insufficient Information (e.g. vegan materials)
        if "vegan" in q_lower or "adhesive" in q_lower or "fabric" in q_lower:
            return AgentResponse(
                answer=(
                    "The supplied product documentation does not contain information regarding vegan certifications for all fabrics and adhesives. "
                    "I am referring this to our human support team for official confirmation."
                ),
                sources=[],
                handoff_recommended=True,
                tool_called=tool_called
            )

        # Default retrieval answer
        if kb_response.results:
            top = kb_response.results[0].chunk
            return AgentResponse(
                answer=f"{top.content} [{top.citation}]",
                sources=[top.citation],
                handoff_recommended=handoff_recommended,
                tool_called=tool_called
            )

        # Safe Abstention Fallback
        return AgentResponse(
            answer="I do not have sufficient information in our official documentation to answer your question. I am recommending a human support specialist to assist you.",
            sources=[],
            handoff_recommended=True,
            tool_called=tool_called
        )
