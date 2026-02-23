# db.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import asyncio
from datetime import datetime, timezone
import hashlib
from typing import List, Dict, Any

load_dotenv()

_client: AsyncIOMotorClient | None = None


def connect_mongo() -> None:
    global _client
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is missing")
    _client = AsyncIOMotorClient(uri)

def close_mongo() -> None:
    global _client
    if _client:
        _client.close()
        _client = None

def get_db():
    if _client is None:
        raise RuntimeError("Mongo client not initialized")
    db_name = os.getenv("MONGODB_DB")
    if not db_name:
        raise RuntimeError("MONGODB_DB is missing")
    return _client[db_name]

def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def _serialize_cart(cart: dict) -> dict:
    return _serialize(cart)

def serialize_cart(cart: dict) -> dict:
    return _serialize(cart)

async def check_wallet_by_id(id: str) -> dict:
    """
    Checks wallet for a user id in the ChatPay DB's `users` collection.

    Expects documents shaped like:
      { "id": "<user_id>", "address": "...", "wallet_id": "..." }

    Returns:
      - {"wallet_id": "...", "address": "..."} on success
      - {"error": "..."} otherwise
    """
    short_id = (id[:6] + "...") if isinstance(id, str) and len(id) > 6 else id
    print(f"[mongo] check_wallet_by_id start id={short_id}")

    users = get_db()["users"]

    record = await users.find_one({"id": id}, {"_id": 0})

    if not record:
        print(f"[mongo] check_wallet_by_id miss id={short_id}")
        return {"error": "No wallet associated with this id"}

    address = record.get("address")
    wallet_id = record.get("wallet_id")

    if address and wallet_id:
        print(f"[mongo] check_wallet_by_id hit id={short_id}")
        return {"wallet_id": wallet_id, "address": address}

    print(
        f"[mongo] check_wallet_by_id incomplete id={short_id} "
        f"has_address={bool(address)} has_wallet_id={bool(wallet_id)}"
    )
    return {"error": "No wallet associated with this id"}

def users():
    return get_db()["users"]

def get_business_db(business_id: str):
    if _client is None:
        raise RuntimeError("Mongo client not initialized")
    if not business_id:
        raise RuntimeError("business_id is required")

    return _client[_business_db_name(business_id)]

def conversations(business_id: str):
    return get_business_db(business_id)["conversations"]


async def append_conversation_message(*, business_id: str, thread_id: str, role: str, content: str):
    await conversations(business_id).update_one(
        {"thread_id": thread_id},
        {
            "$push": {
                "messages": {
                    "role": role,
                    "content": content,
                    "ts": datetime.now(timezone.utc),
                }
            },
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )


async def load_last_conversation_messages(*, business_id: str, thread_id: str, limit: int = 20):
    doc = await conversations(business_id).find_one(
        {"thread_id": thread_id},
        {"messages": {"$slice": -limit}, "_id": 0},
    )
    return doc.get("messages", []) if doc else []

def carts(business_id: str):
    return get_business_db(business_id)["carts"]


async def load_cart_db(*, business_id: str, thread_id: str) -> dict:
    cart = await carts(business_id).find_one(
        {"thread_id": thread_id},
        {"_id": 0},
    )
    return cart or {
        "business_id": business_id,
        "thread_id": thread_id,
        "items": [],
        "summary": {},
        "updated_at": datetime.now(timezone.utc),
    }


async def save_cart_db(cart: dict) -> None:
    cart["updated_at"] = datetime.now(timezone.utc)
    await carts(cart["business_id"]).update_one(
        {"thread_id": cart["thread_id"]},
        {"$set": cart},
        upsert=True,
    )

def _business_db_name(business_id: str) -> str:
    h = hashlib.sha1(business_id.encode()).hexdigest()[:8]
    return f"chatpay_b_{h}"

def payments(business_id: str):
    return get_business_db(business_id)["payments"]

def orders(business_id: str):
    db = get_business_db(business_id)
    return db["orders"]

async def get_categories(business_id: str) -> List[Dict[str, Any]]:
    """
    Returns all categories for a business.
    """
    db: AsyncIOMotorDatabase = get_db()

    cursor = db.categories.find({}, {"_id": 0})
    categories = []

    async for doc in cursor:
        categories.append(doc)

    return categories
'''
if __name__ == "__main__":
    async def _main():
        connect_mongo()
        try:
            out = await get_categories('0b235c20-be85-4f46-831c-d382891a2fa1')
            print(out)
        finally:
            close_mongo()

    asyncio.run(_main())

'''
