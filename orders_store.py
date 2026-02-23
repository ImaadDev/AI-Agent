# orders_store.py
from datetime import datetime, timezone
from typing import Dict, Any
from db import orders   # 👈 same pattern as payments


def _now():
    return datetime.now(timezone.utc)


# CREATE order
async def create_order(state: Dict[str, Any]) -> Dict[str, Any]:
    state["status"] = "pending"  # pending | fulfilled
    state["created_at"] = _now()
    state["updated_at"] = _now()

    res = await orders(state["business_id"]).insert_one(state)
    state["_id"] = res.inserted_id
    return state


# LOAD order
async def load_order(order_id, business_id: str):
    return await orders(business_id).find_one({"_id": order_id})


# MARK fulfilled
async def mark_order_fulfilled(order_id, business_id: str):
    await orders(business_id).update_one(
        {"_id": order_id},
        {
            "$set": {
                "status": "fulfilled",
                "updated_at": _now(),
            }
        },
    )