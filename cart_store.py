# cart_store.py
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from db import load_cart_db, save_cart_db

import os

LOCAL_STORE_DIR = os.getenv("LOCAL_STORE_DIR", "./local_conversations")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cart_path(thread_id: str) -> str:
    _ensure_dir(LOCAL_STORE_DIR)
    safe = "".join(c for c in thread_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(LOCAL_STORE_DIR, f"{safe}.cart.json")


async def load_cart(thread_id: str, business_id: str) -> dict:
    cart = await load_cart_db(
        business_id=business_id,
        thread_id=thread_id,
    )
    return _serialize_cart(cart)



async def save_cart(cart: dict) -> None:
    await save_cart_db(cart)


_PRICE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?\s*$")


def parse_price(price_text: Optional[str]) -> Tuple[float, str]:
    if not price_text:
        return 0.0, "USDC"
    m = _PRICE_RE.match(price_text)
    if not m:
        return 0.0, "USDC"
    amount = float(m.group(1))
    currency = (m.group(2) or "USDC").upper()
    return amount, currency


def _item_key(item: Dict[str, Any]) -> str:
    ref = item.get("product_ref") or {}
    return (
        (ref.get("source_chunk_id") or "")
        + "|"
        + (ref.get("source_file_id") or "")
        + "|"
        + (item.get("name") or "")
    )


def recompute_summary(cart: Dict[str, Any]) -> None:
    items = cart.get("items") or []
    subtotal = 0.0
    count = 0
    currency = cart.get("summary", {}).get("currency") or "USDC"

    for it in items:
        qty = int(it.get("qty") or 0)
        unit = float(it.get("unit_price_amount") or 0.0)
        it["line_total_amount"] = unit * qty
        subtotal += it["line_total_amount"]
        count += qty
        currency = (it.get("unit_price_currency") or currency or "USDC").upper()

    cart["summary"] = {
        "item_count": count,
        "subtotal_amount": subtotal,
        "currency": currency,
    }


def add_item(cart: Dict[str, Any], product: Dict[str, Any], qty: int = 1) -> Dict[str, Any]:
    qty = max(1, int(qty))

    amount, currency = parse_price(product.get("price"))
    new_item = {
        "product_ref": {
            "source_chunk_id": product.get("source_chunk_id"),
            "source_file_id": product.get("source_file_id"),
        },
        "name": product.get("name"),
        "unit_price_text": product.get("price"),
        "unit_price_amount": amount,
        "unit_price_currency": currency,
        "qty": qty,
        "suggested_asset_ids": product.get("suggested_asset_ids") or [],
    }

    items = cart.setdefault("items", [])
    key = _item_key(new_item)

    for it in items:
        if _item_key(it) == key:
            it["qty"] = int(it.get("qty") or 0) + qty
            recompute_summary(cart)
            return cart

    items.append(new_item)
    recompute_summary(cart)
    return cart
def serialize_cart(cart: dict) -> dict:
    if not cart:
        return cart

    out = dict(cart)

    if "updated_at" in out and hasattr(out["updated_at"], "isoformat"):
        out["updated_at"] = out["updated_at"].isoformat()

    return out

def _serialize_cart(cart: dict) -> dict:
    if not cart:
        return cart

    out = dict(cart)

    if "updated_at" in out and hasattr(out["updated_at"], "isoformat"):
        out["updated_at"] = out["updated_at"].isoformat()

    return out

def remove_item(cart: Dict[str, Any], source_chunk_id: str) -> Dict[str, Any]:
    items = cart.get("items") or []
    cart["items"] = [
        it for it in items
        if (it.get("product_ref", {}).get("source_chunk_id") != source_chunk_id)
    ]
    recompute_summary(cart)
    return cart


def set_qty(cart: Dict[str, Any], source_chunk_id: str, qty: int) -> Dict[str, Any]:
    qty = int(qty)
    items = cart.get("items") or []
    for it in items:
        if it.get("product_ref", {}).get("source_chunk_id") == source_chunk_id:
            it["qty"] = max(0, qty)
    cart["items"] = [it for it in items if int(it.get("qty") or 0) > 0]
    recompute_summary(cart)
    return cart


def clear_cart(cart: Dict[str, Any]) -> Dict[str, Any]:
    cart["items"] = []
    recompute_summary(cart)
    return cart
