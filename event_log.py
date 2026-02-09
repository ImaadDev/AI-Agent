from datetime import datetime, timezone
from db import get_db

def log_event(event: dict):
    event["ts"] = datetime.now(timezone.utc)
    get_db()["events"].insert_one(event)