import asyncio
import os
import json

from shopping_agent import chat_turn
from db import connect_mongo, close_mongo
from telegram_formatter import format_for_telegram


async def main():
    connect_mongo()
    try:
        thread_id = os.getenv("THREAD_ID", "test-thread-1")
        business_id = os.getenv("BUSINESS_ID")

        print(f"Thread: {thread_id}\n")

        while True:
            text = input("You: ").strip()
            if text.lower() == "exit":
                break

            reply = await chat_turn(
                thread_id=thread_id,
                text=text,
                business_id=business_id,
            )

            # Parse agent envelope
            try:
                envelope = json.loads(reply)
            except Exception:
                print(f"Agent: {reply}\n")
                continue

            # Format for Telegram
            messages = format_for_telegram(envelope)

            # Simulate Telegram send
            for m in messages:
                if m["type"] == "text":
                    print(f"Agent: {m['content']}\n")
                elif m["type"] == "photo":
                    print(f"Agent [photo]: {m['content']}\n")

    finally:
        close_mongo()


if __name__ == "__main__":
    asyncio.run(main())
