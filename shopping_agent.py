# shopping_agent.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, Literal, TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from rag_query import rag_search, rag_message

from llm_wrapper import call_llm

from event_log import log_event

from cart_store import (
    load_cart,
    save_cart,
    add_item,
    remove_item,
    set_qty,
    clear_cart,
    serialize_cart,
)


from payments_store import create_payment_attempt, load_latest_payment, update_payment_attempt

from solders.keypair import Keypair

from paylink import solana_pay_url, USDC_MINT_MAINNET

import re

from db import load_cart_db, save_cart_db

from db import (
    append_conversation_message,
    load_last_conversation_messages,
)

from geocode import geocode_address, is_allowed_location

load_dotenv()

# ----------------------------
# ENV
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

DEFAULT_BUSINESS_ID = os.getenv("BUSINESS_ID", "UNKNOWN_BUSINESS")
LOCAL_STORE_DIR = os.getenv("LOCAL_STORE_DIR", "./local_conversations")
NOT_READY_TEXT = os.getenv("NOT_READY_TEXT", "no payments yet")

RECENT_ENTITIES: dict[str, list[dict]] = {}
# ----------------------------
# Local conversation storage (JSONL per thread)
# ----------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)




# ----------------------------
# LangGraph State
# ----------------------------
class State(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    business_id: str
    turn_id: str


# ----------------------------
# Router LLM
# ----------------------------
router_llm = ChatOpenAI(model=os.getenv("ROUTER_MODEL", "gpt-4o-mini"), temperature=0)

ROUTER_SYSTEM = """You are a router for an agentic commerce backend.

Choose exactly ONE route for the user's latest message:

ENQUIRY:
- questions about products, catalog, company, what you sell, pricing, details, ingredients, etc.

CART:
- cart_add: user wants to add something to cart (e.g. "add X to my cart", "put X in my cart", "get me X", "I want X", "can you add X")
- cart_remove: user wants to remove an item
- cart_view: user asks to see cart
- cart_clear: user wants to clear cart

PAYMENTS:
- paying/checkout for the cart (e.g. "checkout", "buy my cart", "pay now", "I want to pay")

CHITCHAT:
- greetings/thanks/acknowledgements with no commerce intent

IMPORTANT:
- If user says "buy my cart" / "checkout" / "pay" => payments
- If user says "add X to my cart" => cart_add (NOT payments)

Return ONLY one of:
enquiry, cart_add, cart_remove, cart_view, cart_clear, payments, chitchat
"""


def _router_history(state: State, limit: int = 10) -> str:
    lines = []
    for m in state["messages"][-limit:]:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"ASSISTANT: {m.content}")
    return "\n".join(lines)


async def route_intent(state: State) -> str:
    msgs = state["messages"]
    user_last = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            user_last = m.content or ""
            break

    history = _router_history(state, limit=10)
    router_input = f"CONVERSATION (last 10):\n{history}\n\nLATEST USER MESSAGE:\n{user_last}\n"

    resp = await call_llm(
        router_llm,
        [SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=router_input)],
        business_id=state["business_id"],
        thread_id=state["thread_id"],
        turn_id=state["turn_id"],
        agent_node="router",
    )

    label = (resp.content or "").strip().lower()
    allowed = {"enquiry", "cart_add", "cart_remove", "cart_view", "cart_clear", "payments", "chitchat"}
    return label if label in allowed else "chitchat"


# ----------------------------
# Enquiry agent
# ----------------------------
async def enquiry_agent_node(state: State) -> Dict[str, Any]:
    business_id = state.get("business_id") or DEFAULT_BUSINESS_ID

    question = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            question = (m.content or "").strip()
            break

    search = await rag_search(
        business_id=business_id,
        question=question,
        top_k=int(os.getenv("TOP_K", "6")),
    )

    best = search.get("best_product")
    product_matches = search.get("product_matches", []) or []
    info_matches = search.get("info_matches", []) or []

    def _confidence_from_score(score: float) -> float:
        if score is None:
            return 0.0
        return max(0.0, min(1.0, float(score) / 0.5))

    def _guess_page_or_slide(info_match: Dict[str, Any]) -> Optional[int]:
        desc = (info_match.get("description") or "")
        if "[Page" in desc:
            try:
                i = desc.find("[Page")
                j = desc.find("]", i)
                if j != -1:
                    token = desc[i:j].replace("[", "").replace("]", "").strip()
                    parts = token.split()
                    if len(parts) == 2 and parts[0].lower() == "page":
                        return int(parts[1])
            except Exception:
                pass
        return None

    def _pick_provenance(product: Dict[str, Any], info_matches_: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_chunk_id = product.get("source_chunk_id") or product.get("chunk_id")
        source_file_id = product.get("source_file_id") or product.get("file_id")

        suggested_asset_ids = product.get("suggested_asset_ids")
        if suggested_asset_ids is None:
            suggested_asset_ids = product.get("asset_ids") or []

        page_or_slide = product.get("page_or_slide")
        if page_or_slide is None and info_matches_:
            page_or_slide = _guess_page_or_slide(info_matches_[0])

        if (not source_file_id) and info_matches_:
            source_file_id = info_matches_[0].get("file_id")
        if (not source_chunk_id) and info_matches_:
            source_chunk_id = info_matches_[0].get("chunk_id")

        return {
            "source_chunk_id": source_chunk_id,
            "source_file_id": source_file_id,
            "page_or_slide": page_or_slide,
            "suggested_asset_ids": suggested_asset_ids,
        }

    def _to_product_obj(p: Dict[str, Any]) -> Dict[str, Any]:
        prov = _pick_provenance(p, info_matches)
        return {
            "name": p.get("name"),
            "price": p.get("price"),
            "description": p.get("description"),
            "confidence": 1.0 if (best is not None and p is best) else _confidence_from_score(p.get("score") or 0.0),
            "source_chunk_id": prov["source_chunk_id"],
            "source_file_id": prov["source_file_id"],
            "page_or_slide": prov["page_or_slide"],
            "suggested_asset_ids": prov["suggested_asset_ids"],
        }

    if best:
        data = {"result_type": "product", "product": _to_product_obj(best)}
    else:
        data = {"result_type": "products", "products": [_to_product_obj(p) for p in product_matches]}

    last_messages = []
    for m in state["messages"][-10:]:
        if isinstance(m, HumanMessage):
            last_messages.append({"role": "user", "content": (m.content or "").strip()})
        elif isinstance(m, AIMessage):
            last_messages.append({"role": "assistant", "content": (m.content or "").strip()})

    envelope: Dict[str, Any] = {"type": "enquiry", "message": "", "data": data}
    envelope["message"] = (
        await rag_message(
            business_id=business_id,
            question=question,
            last_messages=last_messages,
        )
    ).get("message", "")

    entities = []

    if best:
        entities.append({
            "source_chunk_id": data["product"]["source_chunk_id"],
            "name": data["product"]["name"],
            "confidence": data["product"]["confidence"],
            "ts": _now().isoformat(),
        })
    else:
        for p in data.get("products", [])[:3]:
            entities.append({
                "source_chunk_id": p["source_chunk_id"],
                "name": p["name"],
                "confidence": p["confidence"],
                "ts": _now().isoformat(),
            })

    RECENT_ENTITIES[state["thread_id"]] = entities

    return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}


# ----------------------------
# Cart action extractor (LLM)
# ----------------------------
cart_llm = ChatOpenAI(model=os.getenv("CART_EXTRACTOR_MODEL", "gpt-4o-mini"), temperature=0)

CART_EXTRACT_SYSTEM = """Extract a cart action from the user's message.

Return ONLY valid JSON (no markdown, no commentary).

Schema:
{
  "intent": "add" | "remove" | "set_qty" | "view" | "clear" | "unknown",
  "query": string | null,      // product name / description for add, or identifier for remove
  "qty": integer | null        // only for add/set_qty
}

Rules:
- If user asks to see the cart -> intent="view"
- If user asks to clear cart -> intent="clear"
- If user asks to add/put/get something in cart -> intent="add" and put the product phrase in "query". Default qty=1 if not specified.
- If user asks to remove something -> intent="remove" and put what to remove in "query"
- If user asks to set quantity -> intent="set_qty" with query=what item, qty=number
- If unclear -> intent="unknown"
"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


async def extract_cart_action(user_text: str) -> Dict[str, Any]:
    prompt = f"USER_MESSAGE:\n{user_text}\n"
    resp = await cart_llm.ainvoke([SystemMessage(content=CART_EXTRACT_SYSTEM), HumanMessage(content=prompt)])
    raw = (resp.content or "").strip()

    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return {"intent": "unknown", "query": None, "qty": None}

    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            return {"intent": "unknown", "query": None, "qty": None}
        intent = (obj.get("intent") or "unknown").strip().lower()
        query = obj.get("query")
        qty = obj.get("qty")
        if intent not in {"add", "remove", "set_qty", "view", "clear", "unknown"}:
            intent = "unknown"
        if qty is not None:
            try:
                qty = int(qty)
            except Exception:
                qty = None
        if isinstance(query, str):
            query = query.strip()
            if not query:
                query = None
        else:
            query = None
        return {"intent": intent, "query": query, "qty": qty}
    except Exception:
        return {"intent": "unknown", "query": None, "qty": None}


def _product_from_search(best: dict | None, info_matches: list[dict]) -> dict | None:
    if not best:
        return None
    source_file_id = info_matches[0].get("file_id") if info_matches else None
    return {
        "name": best.get("name"),
        "price": best.get("price"),
        "description": best.get("description"),
        "confidence": best.get("score"),
        "source_chunk_id": best.get("chunk_id"),
        "source_file_id": source_file_id,
        "page_or_slide": None,
        "suggested_asset_ids": best.get("asset_ids") or [],
    }


def _resolve_cart_chunk_id(cart: dict, query: str) -> str | None:
    """
    Resolve a user remove query to a cart item's product_ref.source_chunk_id.
    Accepts either a raw chunk_id or a product name fragment.
    """
    q = (query or "").strip()
    if not q:
        return None

    # If they pasted a chunk_id, use it directly
    if q.startswith("product:"):
        return q

    ql = q.lower()
    items = (cart or {}).get("items") or []

    # Prefer exact-ish name match
    for it in items:
        name = (it.get("name") or "").strip().lower()
        if name == ql:
            return (it.get("product_ref") or {}).get("source_chunk_id")

    # Fallback: contains match
    for it in items:
        name = (it.get("name") or "").strip().lower()
        if ql in name:
            return (it.get("product_ref") or {}).get("source_chunk_id")

    return None


# ----------------------------
# Cart nodes
# ----------------------------
async def cart_view_node(state: State) -> Dict[str, Any]:
    thread_id = state["thread_id"]
    business_id = state.get("business_id") or DEFAULT_BUSINESS_ID

    cart = await load_cart(thread_id, business_id)
    safe_cart = serialize_cart(cart)

    envelope = {
        "type": "cart",
        "message": "Here’s your cart.",
        "data": {"cart": safe_cart},
    }
    return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}



async def cart_clear_node(state: State) -> Dict[str, Any]:
    thread_id = state["thread_id"]
    business_id = state.get("business_id") or DEFAULT_BUSINESS_ID

    cart = await load_cart(thread_id, business_id)
    clear_cart(cart)
    await save_cart(cart)
    

    safe_cart = serialize_cart(cart)

    envelope = {
        "type": "cart",
        "message": "Cleared your cart.",
        "data": {"cart": safe_cart},
    }
    return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}



async def cart_remove_node(state: State) -> Dict[str, Any]:
    thread_id = state["thread_id"]
    business_id = state.get("business_id") or DEFAULT_BUSINESS_ID

    cart = await load_cart(thread_id, business_id)

    user_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_text = (m.content or "").strip()
            break

    act = await extract_cart_action(user_text)
    target = act.get("query")

    if not target:
        safe_cart = serialize_cart(cart)
        envelope = {
            "type": "cart",
            "message": "What should I remove?",
            "data": {"cart": safe_cart},
        }
        return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}

    chunk_id = _resolve_cart_chunk_id(cart, target)
    if not chunk_id:
        safe_cart = serialize_cart(cart)
        envelope = {
            "type": "cart",
            "message": "I couldn’t find that item in your cart.",
            "data": {"cart": safe_cart},
        }
        return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}

    before = len(cart.get("items") or [])
    remove_item(cart, chunk_id)
    after = len(cart.get("items") or [])

    if after < before:
        await save_cart(cart)
       
        msg = "Removed item."
    else:
        msg = "I couldn’t find that item in your cart."

    safe_cart = serialize_cart(cart)
    envelope = {
        "type": "cart",
        "message": msg,
        "data": {"cart": safe_cart},
    }
    return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}



async def cart_add_node(state: State) -> Dict[str, Any]:
    thread_id = state["thread_id"]
    business_id = state.get("business_id") or DEFAULT_BUSINESS_ID

    cart = await load_cart(thread_id, business_id)

    user_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_text = (m.content or "").strip()
            break

    act = await extract_cart_action(user_text)
    query = act.get("query")
    qty = act.get("qty") or 1

    if not query:
        # ---- fallback to last referenced product ----
        entities = RECENT_ENTITIES.get(thread_id, [])
        if entities:
            query = entities[0]["name"]
        else:
            safe_cart = serialize_cart(cart)
            envelope = {
                "type": "cart",
                "message": "What do you want to add?",
                "data": {"cart": safe_cart},
            }
            return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}

    search = await rag_search(
        business_id=business_id,
        question=query,
        top_k=int(os.getenv("TOP_K", "6")),
    )

    best = search.get("best_product")
    product_matches = search.get("product_matches", []) or []
    info_matches = search.get("info_matches", []) or []

    if not best and product_matches:
        best = product_matches[0]

    product = _product_from_search(best, info_matches)
    if not product or not product.get("name"):
        safe_cart = serialize_cart(cart)
        envelope = {
            "type": "cart",
            "message": "I couldn’t find that product. Try a slightly different name.",
            "data": {"cart": safe_cart},
        }
        return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}

    if isinstance(product.get("price"), (int, float)):
        product["price"] = str(product["price"])
    add_item(cart, product, qty=int(qty))
    await save_cart(cart)
    

    safe_cart = serialize_cart(cart)
    envelope = {
        "type": "cart",
        "message": f"Added {int(qty)} × {product.get('name')} to your cart.",
        "data": {"cart": safe_cart},
    }
    return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}



# ----------------------------
# Payments node (kept as-is from your working version)
# ----------------------------
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


async def payments_agent_node(state: State) -> Dict[str, Any]:
    thread_id = state["thread_id"]
    business_id = state.get("business_id") or DEFAULT_BUSINESS_ID

    cart = await load_cart(thread_id, business_id)
    summary = (cart or {}).get("summary") or {}

    # ---- Guard: empty cart ----
    if not summary.get("item_count") or float(summary.get("subtotal_amount", 0)) <= 0:
        envelope = {
            "type": "payments",
            "message": "Your cart is empty. Add items before checking out.",
            "data": {"cart": serialize_cart(cart)},
        }
        return {"messages": [AIMessage(content=json.dumps(envelope))]}

    # ---- Get latest user message ----
    user_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_text = (m.content or "").strip()
            break

    # ---- Load latest payment attempt (if any) ----
    latest = await load_latest_payment(thread_id, business_id)

    # =====================================================
    # STEP 1: User says checkout → create attempt
    # =====================================================
    # =====================================================
    # STEP 1: User says checkout → create attempt
    # =====================================================
    if user_text.lower() in {"checkout", "buy", "buy my cart", "pay", "pay now"}:

        payment = await create_payment_attempt({
            "business_id": business_id,
            "thread_id": thread_id,
            "stage": "awaiting_email",
            "cart_snapshot": serialize_cart(cart),
        })

        envelope = {
            "type": "payments",
            "message": "What’s your email?",
            "data": {"payment_id": str(payment["_id"])},
        }
        return {"messages": [AIMessage(content=json.dumps(envelope))]}


    # =====================================================
    # STEP 2: Email entered
    # =====================================================
    if latest and latest.get("stage") == "awaiting_email" and _EMAIL_RE.match(user_text):

        await update_payment_attempt(
            latest["_id"],
            business_id,
            {
                "stage": "awaiting_address",
                "email": user_text,
            },
        )

        envelope = {
            "type": "payments",
            "message": "What’s your delivery address?",
            "data": None,
        }
        return {"messages": [AIMessage(content=json.dumps(envelope))]}


    # =====================================================
    # STEP 3: Address entered
    # =====================================================
    if latest and latest.get("stage") == "awaiting_address":

        geo = await geocode_address(user_text)
        allowed, reason = is_allowed_location(geo)

        if not allowed:
            envelope = {
                "type": "payments",
                "message": reason,
                "data": None,
            }
            return {"messages": [AIMessage(content=json.dumps(envelope))]}

        await update_payment_attempt(
            latest["_id"],
            business_id,
            {
                "stage": "address_verified",
                "address": user_text,
                "geo": geo,
            },
        )

        envelope = {
            "type": "payments",
            "message": "Payments not configured yet.",
            "data": None,
        }
        return {"messages": [AIMessage(content=json.dumps(envelope))]}



# ----------------------------
# Chitchat node
# ----------------------------
chitchat_message_llm = ChatOpenAI(model=os.getenv("CHITCHAT_MODEL", "gpt-4o-mini"), temperature=0.6)

CHITCHAT_SYSTEM = """You are the UI narration for a commerce assistant.

The user is not asking about products, cart, or payments.
Respond naturally to the user's message.

Rules:
- Output plain text only.
- 1–3 sentences.
- Friendly, human, not repetitive.
- Do NOT mention products, cart, or payments unless the user asks.
- If the user greets, greet back.
"""


async def chitchat_agent_node(state: State) -> Dict[str, Any]:
    user_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_text = (m.content or "").strip()
            break

    history_snips = []
    for m in state["messages"][-6:]:
        if isinstance(m, HumanMessage):
            history_snips.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            history_snips.append(f"ASSISTANT: {m.content}")

    prompt = f"CONVERSATION (recent):\n" + "\n".join(history_snips) + f"\n\nUSER MESSAGE:\n{user_text}\n"
    resp = await chitchat_message_llm.ainvoke([SystemMessage(content=CHITCHAT_SYSTEM), HumanMessage(content=prompt)])
    msg = " ".join((resp.content or "").strip().split()) or "Hey — what can I help you find today?"

    envelope = {"type": "chitchat", "message": msg, "data": None}
    return {"messages": [AIMessage(content=json.dumps(envelope, ensure_ascii=False))]}


# ----------------------------
# Graph wiring
# ----------------------------
builder = StateGraph(State)
builder.add_node("enquiry", enquiry_agent_node)
builder.add_node("cart_add", cart_add_node)
builder.add_node("cart_remove", cart_remove_node)
builder.add_node("cart_view", cart_view_node)
builder.add_node("cart_clear", cart_clear_node)
builder.add_node("payments", payments_agent_node)
builder.add_node("chitchat", chitchat_agent_node)


async def _route(state: State) -> str:
    return await route_intent(state)


builder.add_conditional_edges(
    START,
    _route,
    {
        "enquiry": "enquiry",
        "cart_add": "cart_add",
        "cart_remove": "cart_remove",
        "cart_view": "cart_view",
        "cart_clear": "cart_clear",
        "payments": "payments",
        "chitchat": "chitchat",
    },
)

builder.add_edge("enquiry", END)
builder.add_edge("cart_add", END)
builder.add_edge("cart_remove", END)
builder.add_edge("cart_view", END)
builder.add_edge("cart_clear", END)
builder.add_edge("payments", END)
builder.add_edge("chitchat", END)

graph = builder.compile()


# ----------------------------
# Public entrypoint (CLI calls this)
# ----------------------------
async def chat_turn(
    *,
    thread_id: str,
    text: str,
    turn_id: str,
    business_id: Optional[str] = None,
    history_limit: int = 20,
) -> str:
    business_id = business_id or DEFAULT_BUSINESS_ID

    await append_conversation_message(
        business_id=business_id,
        thread_id=thread_id,
        role="user",
        content=text,
    )

    last = await load_last_conversation_messages(
        business_id=business_id,
        thread_id=thread_id,
        limit=history_limit,
    )
    msgs: List[BaseMessage] = []
    for m in last:
        if m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
        else:
            msgs.append(HumanMessage(content=m["content"]))

    out = await graph.ainvoke(
        {"thread_id": thread_id, "business_id": business_id, "messages": msgs,"turn_id": turn_id},
        config={"configurable": {"thread_id": thread_id}},
    )

    final_text = out["messages"][-1].content if out.get("messages") else ""
    await append_conversation_message(
        business_id=business_id,
        thread_id=thread_id,
        role="assistant",
        content=final_text,
    )
    return final_text


