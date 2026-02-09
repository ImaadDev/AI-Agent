# telegram_formatter.py

from typing import Dict, Any, List


CONFIDENCE_THRESHOLD = 0.5


def format_for_telegram(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert agent envelope JSON into Telegram-friendly messages.

    Returns a list of actions:
    {
      "type": "text" | "photo",
      "content": str
    }
    """

    out: List[Dict[str, Any]] = []
    etype = envelope.get("type")

    # -----------------
    # CHITCHAT
    # -----------------
    if etype == "chitchat":
        out.append({
            "type": "text",
            "content": envelope.get("message", "")
        })
        return out

    # -----------------
    # PAYMENTS (disabled)
    # -----------------
    if etype == "payments":
        out.append({
            "type": "text",
            "content": "Payments not configured yet."
        })
        return out

    # -----------------
    # CART (NO IMAGES)
    # -----------------
    if etype == "cart":
        cart = (envelope.get("data") or {}).get("cart") or {}
        items = cart.get("items") or []
        summary = cart.get("summary") or {}

        if not items:
            out.append({
                "type": "text",
                "content": "🛒 Your cart is empty."
            })
            return out

        lines = ["🛒 *Your cart*\n"]
        for it in items:
            name = it.get("name")
            qty = it.get("qty")
            price = it.get("unit_price_amount")
            currency = it.get("unit_price_currency", "")
            lines.append(f"• {name}\n  Qty: {qty} · {price} {currency}")

        lines.append(
            f"\nSubtotal: {summary.get('subtotal_amount')} {summary.get('currency')}"
        )

        out.append({
            "type": "text",
            "content": "\n".join(lines)
        })
        return out

    # -----------------
    # ENQUIRY
    # -----------------
    if etype == "enquiry":
        data = envelope.get("data") or {}
        if "product" in data and data["product"]:
            products = [data["product"]]
        else:
            products = data.get("products") or []

        # 1. Images first (one per product)
        for p in products:
            confidence = float(p.get("confidence") or 0.0)
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            assets = p.get("suggested_asset_ids") or []
            if assets:
                out.append({
                    "type": "photo",
                    "content": assets[0],  # one image per product
                })

        # 2. Then the text message
        out.append({
            "type": "text",
            "content": envelope.get("message", "")
        })

        return out

    # -----------------
    # FALLBACK
    # -----------------
    out.append({
        "type": "text",
        "content": envelope.get("message", "")
    })
    return out
