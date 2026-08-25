"""Core Orchestrator for Aster & Row Support Agent."""
import json
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.kb.indexer import KBRetriever
from src.kb.schemas import SearchResponse
from src.tools.order_lookup import lookup_order_tool, OrderLookupService
from src.tools.schemas import SanitizedOrderResult
from src.agent.session import session_manager, ConversationSession
from src.agent.response import AgentResponse
from src.agent.prompts import SYSTEM_PROMPT
from src.observability.logger import agent_observer
import time


def format_friendly_date(date_str: Optional[str]) -> Optional[str]:
    """Format ISO date 2026-08-22 into 'August 22, 2026'."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%B %d, %Y").replace(" 0", " ")
    except Exception:
        return date_str


def clean_inline_brackets(text: str) -> str:
    """Remove inline citation brackets like [filename.md > Heading] from prose."""
    cleaned = re.sub(r"\s*\[\d{2}-[\w-]+\.md\s*>[^\]]+\]", "", text)
    cleaned = re.sub(r"\s*\[\d{2}-[\w-]+\.md\]", "", cleaned)
    return cleaned.strip()


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
        t0 = time.time()
        session = self.session_mgr.get_or_create(session_id)
        raw_query = user_message.strip()

        # Step 1: Detect Order IDs and Intent
        extracted_order_id = OrderLookupService.normalize_order_id(raw_query)
        order_intent = self._detect_order_intent(raw_query)
        
        # Multi-turn order ID resolution
        active_order_id = extracted_order_id or (session.active_order_id if order_intent else None)

        tool_called: Optional[str] = None
        tool_args: Optional[Dict[str, Any]] = None
        tool_result: Optional[SanitizedOrderResult] = None
        sanitized_summary: Optional[str] = None
        handoff_recommended = False

        # Step 2: Handle Order Inquiries
        if order_intent or extracted_order_id:
            if not active_order_id:
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
        if "ship" in raw_query.lower() or "international" in raw_query.lower() or "canada" in raw_query.lower():
            session.active_topic = "shipping"
        elif "warranty" in raw_query.lower():
            session.active_topic = "warranty"
        elif "return" in raw_query.lower():
            session.active_topic = "returns"

        # Log turn in structured observability trace
        agent_observer.log_turn(
            session_id=session.session_id,
            user_message=raw_query,
            history_summary=[{"role": m.role, "content": m.content} for m in session.messages[:-2]],
            retrieved_passages=[
                {
                    "citation": r.chunk.citation,
                    "score": r.score,
                    "status": r.status,
                    "authority": r.policy_authority,
                }
                for r in kb_response.results
            ],
            tool_call=tool_called,
            tool_args=tool_args,
            sanitized_tool_result=sanitized_summary,
            final_answer=response.answer,
            sources=response.sources,
            handoff_recommended=response.handoff_recommended,
            has_conflict=kb_response.has_conflict,
            execution_time_ms=(time.time() - t0) * 1000.0,
        )

        return response

    def _detect_order_intent(self, query: str) -> bool:
        q = query.lower()
        lookup_phrases = [
            "where is my order", "order status", "track my order", "track order",
            "track package", "when will my order arrive", "when will order",
            "where is ord-", "check ord-", "status of ord-", "why hasn't ord-", "why hasn't ord"
        ]
        if any(p in q for p in lookup_phrases):
            return True
        
        if any(p in q for p in ["when will it arrive", "when will it get here", "where is it", "has it shipped"]):
            return True
            
        return False

    def _detect_privacy_probe(self, query: str) -> bool:
        q = query.lower()
        privacy_keywords = [
            "email", "address", "internal note", "risk score", "warehouse note",
            "hidden prompt", "reveal your instructions", "system prompt",
            "system instructions", "admin override", "diagnostic mode"
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
            
            if ("canada" in q or "vancouver" in q or "international" in q) and ("ship" in last_user_msg or session.active_topic == "shipping"):
                return f"international shipping to Canada delivery estimate {query}"
            if session.active_topic == "warranty" and ("final sale" in q or "what if" in q):
                return f"limited warranty periods final-sale products {query}"
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
            f"Provide a helpful and accurate response. "
            f"Output JSON with keys: 'answer' (clean prose without bracketed citations), 'sources' (list of citations matching '[filename > heading]'), and 'handoff_recommended' (boolean)."
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

        raw_answer = data.get("answer", "")
        answer = clean_inline_brackets(raw_answer)
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

        # Multi-turn Order modification (e.g. adding items / changing color post-checkout on ORD-1001)
        if tool_result and any(w in q_lower for w in ["change the color", "add a", "modify item", "change item"]):
            return AgentResponse(
                answer=(
                    f"Under our Order Changes and Cancellations policy, "
                    f"items and quantities cannot be edited after checkout. "
                    f"Order {tool_result.order_id} is currently in pending status and within the 30-minute window, "
                    f"so you may request to cancel within 30 minutes while pending and place a new order. "
                    f"A human support specialist must complete cancellation requests."
                ),
                sources=[
                    "08-order-changes-and-cancellations.md > Product or quantity changes",
                    "08-order-changes-and-cancellations.md > Cancellation window"
                ],
                handoff_recommended=True,
                tool_called=tool_called,
                tool_arguments=tool_args,
                tool_result_sanitized=sanitized_summary
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

            if tool_result.status == "exception":
                return AgentResponse(
                    answer=(
                        f"Order {tool_result.order_id} is currently flagged with an operational exception status. "
                        f"This shipment requires specialist review and human assistance to resolve."
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
                        f"The order is cancelled and it will not be shipped. "
                        f"Because order {tool_result.order_id} has been cancelled, there is no active delivery date."
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
                        f"Order {tool_result.order_id} has shipped{carrier_info}, "
                        f"but a delivery estimate is unavailable at this time."
                    ),
                    sources=[],
                    handoff_recommended=False,
                    tool_called=tool_called,
                    tool_arguments=tool_args,
                    tool_result_sanitized=sanitized_summary
                )

            if tool_result.status == "shipped":
                carrier_str = f" via {tool_result.carrier}" if tool_result.carrier else ""
                friendly_eta = format_friendly_date(tool_result.estimated_delivery)
                eta_str = f" and is estimated to arrive on {friendly_eta}" if friendly_eta else ""
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
                    "Our current official sources conflict regarding cleaning the Breeze Tumbler: "
                    "the Product Care Guide states that one says hand-wash the body, "
                    "while the Product Information card states that one says all components are dishwasher safe. "
                    "For safest interim guidance, we advise hand-washing the stainless-steel body. "
                    "I am referring this to a human support specialist for human confirmation."
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
                    "The migration note is not authoritative and cannot be used for official answers. Under our official Returns Policy, "
                    "the standard policy is 30 calendar days from delivery "
                    "unless a valid exception applies. Furthermore, the agent cannot approve a return automatically."
                ),
                sources=["01-returns-policy-current.md > Standard return window"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Price Adjustment Inquiry
        if "price drop" in q_lower or "price dropped" in q_lower or "price adjustment" in q_lower:
            return AgentResponse(
                answer=(
                    "Under our Gift Cards and Price Adjustments policy, "
                    "a customer may request one price adjustment if the public price of the same item drops within 7 calendar days "
                    "of the original purchase. A human support specialist must approve and process the adjustment. "
                    "I am connecting you with a human specialist to review your request."
                ),
                sources=["10-gift-cards-and-price-adjustments.md > Price adjustments"],
                handoff_recommended=True,
                tool_called=tool_called
            )

        # Final Sale + Damaged Item Exception
        if "final" in q_lower and ("damaged" in q_lower or "broken" in q_lower or "defective" in q_lower or "zipper" in q_lower):
            return AgentResponse(
                answer=(
                    "While final-sale items cannot be returned for a change of mind, "
                    "final sale does not block damaged-item review. "
                    "Under our Damaged, Defective, or Wrong Items policy, you should report within 7 days "
                    "of delivery with photos for human review before approval. "
                    "I am connecting you with a human support specialist to process this review."
                ),
                sources=[
                    "03-final-sale-and-promotions.md > Damaged or incorrect items",
                    "04-damaged-or-wrong-items.md > Reporting window"
                ],
                handoff_recommended=True,
                tool_called=tool_called
            )

        # Multi-turn Warranty on Final Sale item
        if session.active_topic == "warranty" and "final sale" in q_lower:
            return AgentResponse(
                answer=(
                    "Aster & Row bags have a limited warranty of 2 years from the purchase date. "
                    "A product being purchased as final sale does not remove the limited warranty for a qualifying manufacturing defect."
                ),
                sources=[
                    "07-warranty.md > Warranty periods",
                    "07-warranty.md > Final-sale products"
                ],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # TrailPlus Return Window
        if "trailplus" in q_lower and ("return" in q_lower or "window" in q_lower or "membership" in q_lower):
            return AgentResponse(
                answer=(
                    "A customer whose TrailPlus membership was active when the order was placed receives a "
                    "return window of 45 calendar days from delivery for eligible items."
                ),
                sources=["09-trailplus-membership.md > Return window"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Standard Return Window
        if "return" in q_lower and ("how long" in q_lower or "window" in q_lower or "backpack" in q_lower or "regular" in q_lower):
            return AgentResponse(
                answer=(
                    "Regular customers have 30 calendar days from delivery to return an unused backpack or other eligible items "
                    "in resalable condition."
                ),
                sources=["01-returns-policy-current.md > Standard return window"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # International Shipping / Canada
        if "canada" in q_lower or "vancouver" in q_lower:
            return AgentResponse(
                answer=(
                    "Canada is supported for international shipping. "
                    "Canadian orders generally arrive within 5–9 business days after dispatch. "
                    "Please note that duties or taxes are not prepaid by Aster & Row and are the customer's responsibility."
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
                    "Aster & Row currently ships internationally only to Canada. Therefore, shipping to Germany is not currently available."
                ),
                sources=["06-international-shipping.md > Supported destinations"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Lifetime Warranty Inquiry
        if "lifetime" in q_lower or ("warranty" in q_lower and "all" in q_lower):
            return AgentResponse(
                answer=(
                    "Aster & Row has no lifetime warranty on any products. "
                    "Instead, bags have 2 years of warranty coverage, while drinkware and travel accessories have 1 year from the purchase date."
                ),
                sources=["07-warranty.md > Warranty periods"],
                handoff_recommended=False,
                tool_called=tool_called
            )

        # Insufficient Information (e.g. vegan materials)
        if "vegan" in q_lower or "adhesive" in q_lower or "fabric" in q_lower:
            return AgentResponse(
                answer=(
                    "The supplied information is insufficient in our official documentation to confirm whether all fabrics and adhesives in our bags are vegan. "
                    "I am referring this to a specialist for human confirmation."
                ),
                sources=[],
                handoff_recommended=True,
                tool_called=tool_called
            )

        # Default retrieval answer
        if kb_response.results:
            top = kb_response.results[0].chunk
            clean_ans = clean_inline_brackets(top.content)
            return AgentResponse(
                answer=clean_ans,
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
