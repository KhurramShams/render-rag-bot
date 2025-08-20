import redis
import zlib, json, base64
import os
from dotenv import load_dotenv

load_dotenv()
redis_url = os.getenv("REDIS_URL")

redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

SESSION_EXPIRY_SECONDS = 300  # 5 minutes

def save_chat_history(session_id: str, role: str, content: str):
    key = f"chat_history:{session_id}"
    # JSON → compress → base64 encode
    raw = json.dumps({"role": role, "content": content}).encode("utf-8")
    compressed = zlib.compress(raw)
    safe_value = base64.b64encode(compressed).decode("utf-8")

    redis_client.rpush(key, safe_value)
    redis_client.expire(key, SESSION_EXPIRY_SECONDS)


def get_chat_history(session_id: str):
    key = f"chat_history:{session_id}"
    encoded_messages = redis_client.lrange(key, 0, -1)

    history = []
    for msg in encoded_messages:
        if isinstance(msg, bytes):
            msg = msg.decode("utf-8")
        try:
            compressed = base64.b64decode(msg.encode("utf-8"))
            decompressed = zlib.decompress(compressed).decode("utf-8")
            history.append(json.loads(decompressed))
        except Exception as e:
            print(f"Error decoding message: {e}")
            continue

    return history
