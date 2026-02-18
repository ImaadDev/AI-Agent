import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

ALLOWED_COUNTRIES = {"Ireland", "United Kingdom", "Germany"}


async def geocode_address(address: str) -> dict | None:
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {
        "address": address,
        "key": GOOGLE_API_KEY,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)

    if resp.status_code != 200:
        return None

    data = resp.json()

    if not data.get("results"):
        return None

    result = data["results"][0]
    components = result.get("address_components", [])

    country = None
    city = None

    for comp in components:
        if "country" in comp["types"]:
            country = comp["long_name"]
        if "locality" in comp["types"]:
            city = comp["long_name"]

    return {
        "country": country,
        "city": city,
        "formatted_address": result.get("formatted_address"),
    }


def is_allowed_location(geo: dict) -> tuple[bool, str]:
    if not geo:
        return False, "Address could not be verified."

    if geo["country"] not in ALLOWED_COUNTRIES:
        return False, "We don't ship to that country."

    return True, "OK"


# -----------------------
# TEST BLOCK
# -----------------------
'''
if __name__ == "__main__":
    import asyncio

    async def test():
        address = "Frazer Town bangalore"
        geo = await geocode_address(address)
        print("GEOCODE RESULT:", geo)

        allowed, reason = is_allowed_location(geo)
        print("ALLOWED:", allowed)
        print("REASON:", reason)

    asyncio.run(test())
'''