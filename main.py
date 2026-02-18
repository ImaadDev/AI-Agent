import os
import time
import json
import redis
from shopping_agent import chat_turn
import asyncio
from db import connect_mongo, close_mongo, get_categories
from telegram_formatter import format_for_telegram
import httpx
import uuid

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
WORKER_ID = os.getenv("WORKER_ID", "worker-1")
business_id = os.getenv("BUSINESS_ID")

TELEGRAM_SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# Global keyboard (loaded once)
REPLY_KEYBOARD = None


async def build_keyboard():
    global REPLY_KEYBOARD

    categories = await get_categories(business_id)

    keyboard = [
        [{"text": c["title"]}]
        for c in categories
        if "title" in c
    ]

    # Optional static buttons
    keyboard.append([{"text": "🛒 View Cart"}])
    keyboard.append([{"text": "💳 Checkout"}])

    REPLY_KEYBOARD = {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": False
    }


async def send_telegram_message(chat_id: int, msg: dict, worker_id: str, trace_id: str):
    """
    msg = { "type": "text" | "photo", "content": str }
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:

            if msg["type"] == "text":
                url = TELEGRAM_SEND_URL
                payload = {
                    "chat_id": chat_id,
                    "text": msg["content"],
                    "parse_mode": "Markdown",
                    "reply_markup": REPLY_KEYBOARD
                }
                if "inline_buttons" in msg:
                    payload["reply_markup"] = {
                        "inline_keyboard": msg["inline_buttons"]
                    }
                else:
                    payload["reply_markup"] = REPLY_KEYBOARD

            elif msg["type"] == "photo":
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
                payload = {
                    "chat_id": chat_id,
                    "photo": msg["content"],
                }

            else:
                return

            resp = await client.post(url, json=payload)

            if resp.status_code >= 400:
                print(
                    f"[worker {worker_id}] ERR_TELEGRAM_HTTP trace_id={trace_id} "
                    f"chat_id={chat_id} status={resp.status_code} body={resp.text[:200]}"
                )

    except Exception as e:
        print(
            f"[worker {worker_id}] ERR_SEND_TELEGRAM trace_id={trace_id} "
            f"chat_id={chat_id} err={repr(e)}"
        )
        raise


async def main():
    print(f"[{WORKER_ID}] started")
    print(f"[{WORKER_ID}] listening on chatpay_queue")

    connect_mongo()
    await build_keyboard()

    try:
        while True:
            task = r.blpop("chatpay_queue", timeout=1)

            if not task:
                continue

            _, raw = task
            print(f"\n[{WORKER_ID}] raw message:")

            try:
                data = json.loads(raw)

                callback = data.get("callback_query")

                #  Handle Button Click
                if callback:
                    print("Callback:", callback)

                    user_id = callback["from"]["id"]
                    user_text = callback.get("data")
                    chat_id = callback["message"]["chat"]["id"]

                #  Handle Normal Message
                else:
                    message = data.get("message") or {}
                    if not message:
                        continue  

                    sender = message.get("from") or {}
                    user_id = sender.get("id")
                    user_text = (message.get("text") or "").strip()
                    chat = message.get("chat") or {}
                    chat_id = chat.get("id")

                thread_id = str(user_id)

                print(f"[{WORKER_ID}] thread_id={thread_id}")
                print(f"[{WORKER_ID}] text={user_text}")

                turn_id = str(uuid.uuid4())

                response = await chat_turn(
                    thread_id=thread_id,
                    text=user_text,
                    business_id=business_id,
                    turn_id=turn_id
                )

                print(f"[{WORKER_ID}] agent_response={response}")

                envelope = json.loads(response)
                messages = format_for_telegram(envelope)

                for msg in messages:
                    await send_telegram_message(
                        chat_id=chat_id,
                        msg=msg,
                        worker_id=WORKER_ID,
                        trace_id=str(data.get("update_id"))
                    )

            except Exception as e:
                print(f"[{WORKER_ID}] ERROR: {e}")

    finally:
        print(f"[{WORKER_ID}] shutting down")
        close_mongo()

if __name__ == "__main__":
    asyncio.run(main())