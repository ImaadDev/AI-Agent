# rag_query.py
import os
import math
from typing import List, Dict, Any

from openai import OpenAI
from db import get_db

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _embed_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0

def _fmt_last_messages(last_messages: List[Dict[str, str]], limit: int = 10) -> str:
    msgs = last_messages[-limit:] if limit else last_messages
    lines = []
    for m in msgs:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)

async def rag_search(
    *,
    business_id: str,
    question: str,
    top_k: int = 6,
) -> Dict[str, Any]:
    db = get_db()

    # 1) embed query
    q_emb = client.embeddings.create(
        model=_embed_model(),
        input=[question],
    ).data[0].embedding

    # 2) fetch vectors for this business (now includes typed metadata)
    projection = {
        "embedding": 1,
        "type": 1,
        "kind": 1,
        "chunk_id": 1,
        "file_id": 1,
        "source_type": 1,
        "chunk_index": 1,
        "title": 1,
        "text": 1,
        "name": 1,
        "price": 1,
        "description": 1,
        "asset_ids": 1,
        "product_index": 1,
    }

    cursor = db["vectors"].find({"business_id": business_id}, projection)
    vecs = await cursor.to_list(length=20_000)

    scored = []
    for v in vecs:
        emb = v.get("embedding")
        if not emb:
            continue
        score = _cosine(q_emb, emb)
        scored.append((score, v))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, min(top_k, 50))]

    product_matches = []
    info_matches = []

    for score, v in top:
        vtype = v.get("type") or ("product" if v.get("kind") == "product" else "info")

        if vtype == "product":
            
            product_matches.append(
                {
                    "type": "product",
                    "score": score,
                    "product_index": v.get("product_index"),
                    "name": v.get("name"),
                    "price": v.get("price"),
                    "description": v.get("description"),
                    "asset_ids": v.get("asset_ids", []),
                    "chunk_id": v.get("chunk_id"),
                }
            )
        else:
            info_matches.append(
                {
                    "type": "info",
                    "score": score,
                    "title": v.get("title") or "Info",
                    "description": v.get("text"),
                    "chunk_id": v.get("chunk_id"),
                    "file_id": v.get("file_id"),
                    "source_type": v.get("source_type"),
                    "chunk_index": v.get("chunk_index"),
                }
            )

    # Pick best single product if it's clearly the winner
    best_product = None
    if product_matches:
        top1 = product_matches[0]
        top2 = product_matches[1] if len(product_matches) > 1 else None

        top1_score = top1["score"]
        top2_score = top2["score"] if top2 else 0.0
        gap = top1_score - top2_score

        # Decision rule (tune later)
        if top1_score >= 0.45 and (gap >= 0.05 or top1_score >= 0.60):
            best_product = top1

    return {
        "question": question,
        "best_product": best_product,
        "product_matches": product_matches,
        "info_matches": info_matches,
    }


async def rag_answer(
    *,
    business_id: str,
    question: str,
    top_k: int = 6,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    db = get_db()

    search = await rag_search(business_id=business_id, question=question, top_k=top_k)

    best_product = search.get("best_product")
    product_matches = search["product_matches"]
    info_matches = search["info_matches"]

    print("Search:",search)

    # pull agent config (optional)
    cfg = await db["agent_config"].find_one({"_id": business_id}) or {}
    tone = cfg.get("tone") or "helpful"
    do_rules = cfg.get("do_instructions") or ""
    dont_rules = cfg.get("dont_instructions") or ""

    # Build context from BOTH types
    ctx_parts = []

    for i, m in enumerate(product_matches, start=1):
        ctx_parts.append(
            f"[Product {i} | score={m['score']:.3f}] "
            f"{m.get('name')} | {m.get('price')}\n{m.get('description')}"
        )

    for i, m in enumerate(info_matches, start=1):
        ctx_parts.append(
            f"[Info {i} | score={m['score']:.3f}] "
            f"{m.get('title')}\n{m.get('description')}"
        )

    context = "\n\n".join(ctx_parts)

    prompt = (
        f"You are a store assistant. Tone: {tone}.\n"
        f"Do:\n{do_rules}\n\n"
        f"Don't:\n{dont_rules}\n\n"
        "Use ONLY the provided context. If the answer isn't in the context, say you don't know.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}"
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    output_text = getattr(resp, "output_text", None)
    if not output_text:
        output_text = resp.output[0].content[0].text

    return {
        "question": question,
        "answer": output_text,
        "best_product": best_product,
        "product_matches": product_matches,
        "info_matches": info_matches,
    }

async def rag_chat_answer(
    *,
    business_id: str,
    question: str,
    last_messages: List[Dict[str, str]],  # [{"role":"user"|"assistant", "content":"..."}]
    top_k: int = 6,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    # 1) retrieve (no LLM)
    search = await rag_search(business_id=business_id, question=question, top_k=top_k)

    best_product = search.get("best_product")
    product_matches = search["product_matches"]
    info_matches = search["info_matches"]

    # 2) load agent config (optional)
    db = get_db()
    cfg = await db["agent_config"].find_one({"_id": business_id}) or {}
    tone = cfg.get("tone") or "helpful"
    do_rules = cfg.get("do_instructions") or ""
    dont_rules = cfg.get("dont_instructions") or ""

    # 3) build context
    ctx_parts = []
    for i, m in enumerate(product_matches, start=1):
        ctx_parts.append(
            f"[Product {i} | score={m['score']:.3f}] {m.get('name')} | {m.get('price')}\n{m.get('description')}"
        )
    for i, m in enumerate(info_matches, start=1):
        ctx_parts.append(
            f"[Info {i} | score={m['score']:.3f}] {m.get('title')}\n{m.get('description')}"
        )
    context = "\n\n".join(ctx_parts)

    convo = _fmt_last_messages(last_messages, limit=10)

    prompt = (
        f"You are a store assistant. Tone: {tone}.\n"
        f"Do:\n{do_rules}\n\n"
        f"Don't:\n{dont_rules}\n\n"
        "Use ONLY the provided CONTEXT for factual claims about the store/products.\n"
        "Use CONVERSATION HISTORY only to understand what the user means by follow-ups (e.g., 'tell me more').\n"
        "If the answer isn't in CONTEXT, say you don't know.\n\n"
        f"CONVERSATION HISTORY (last 10):\n{convo}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}"
    )

    resp = client.responses.create(model=model, input=prompt)
    output_text = getattr(resp, "output_text", None) or resp.output[0].content[0].text

    return {
        "question": question,
        "answer": output_text,
        "best_product": best_product,
        "product_matches": product_matches,
        "info_matches": info_matches,
    }

async def rag_message(
    *,
    business_id: str,
    question: str,
    last_messages: List[Dict[str, str]] | None = None,  # NEW
    top_k: int = 6,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Returns a single grounded message string based ONLY on retrieved context.
    Uses conversation history only to resolve follow-ups/ambiguity.
    """
    db = get_db()

    search = await rag_search(business_id=business_id, question=question, top_k=top_k)

    product_matches = search.get("product_matches", []) or []
    info_matches = search.get("info_matches", []) or []

    cfg = await db["agent_config"].find_one({"_id": business_id}) or {}
    tone = cfg.get("tone") or "helpful"
    do_rules = cfg.get("do_instructions") or ""
    dont_rules = cfg.get("dont_instructions") or ""

    ctx_parts = []
    for i, m in enumerate(product_matches, start=1):
        ctx_parts.append(
            f"[Product {i} | score={m['score']:.3f}] "
            f"{m.get('name')} | {m.get('price')}\n{m.get('description')}"
        )
    for i, m in enumerate(info_matches, start=1):
        ctx_parts.append(
            f"[Info {i} | score={m['score']:.3f}] "
            f"{m.get('title')}\n{m.get('description')}"
        )
    context = "\n\n".join(ctx_parts)

    convo = _fmt_last_messages(last_messages or [], limit=10)  # NEW

    prompt = (
        f"You are a store assistant. Tone: {tone}.\n"
        f"Do:\n{do_rules}\n\n"
        f"Don't:\n{dont_rules}\n\n"
        "Rules:\n"
        "- Use ONLY the provided CONTEXT for factual claims.\n"
        "- Use CONVERSATION HISTORY only to understand follow-ups (e.g., 'that one', 'tell me more').\n"
        "- If the answer isn't in CONTEXT, say you don't know.\n"
        "- Be concise.\n\n"
        f"CONVERSATION HISTORY (last 10):\n{convo}\n\n"   # NEW
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}"
    )

    resp = client.responses.create(model=model, input=prompt)

    output_text = getattr(resp, "output_text", None)
    if not output_text:
        output_text = resp.output[0].content[0].text

    return {"message": output_text}

'''
if __name__ == "__main__":
    import asyncio
    from db import connect_mongo, close_mongo

    async def _main():
        connect_mongo()
        try:
            business_id = "a9c45c68-b370-4a5b-949f-1ee4f83854f2"
            question = "What do you sell?"

            out = await rag_search(
                business_id=business_id,
                question=question,
                top_k=int(os.getenv("TOP_K", "6")),
            )
            print(out)
        finally:
            close_mongo()

    asyncio.run(_main())
'''
