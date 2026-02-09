# payments_store.py
from datetime import datetime, timezone
from typing import Dict, Any
from db import payments


def _now():
    return datetime.now(timezone.utc)


# CREATE a new payment attempt
async def create_payment_attempt(state: Dict[str, Any]) -> Dict[str, Any]:
    state["created_at"] = _now()
    state["updated_at"] = _now()
    res = await payments(state["business_id"]).insert_one(state)
    state["_id"] = res.inserted_id
    return state


# LOAD latest attempt (for display / resume)
async def load_latest_payment(thread_id: str, business_id: str) -> Dict[str, Any] | None:
    return await payments(business_id).find_one(
        {"thread_id": thread_id},
        sort=[("created_at", -1)],
        projection={"_id": 1, "stage": 1},
    )


# UPDATE a specific attempt
async def update_payment_attempt(payment_id, business_id: str, updates: Dict[str, Any]) -> None:
    updates["updated_at"] = _now()
    await payments(business_id).update_one(
        {"_id": payment_id},
        {"$set": updates},
    )
