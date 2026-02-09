
from db import users,connect_mongo,close_mongo
import os
from dotenv import load_dotenv
import base64
from requests.exceptions import RequestException
import requests
from solders.pubkey import Pubkey
from spl.token.instructions import (
    get_associated_token_address,
    create_associated_token_account,
    transfer,
    TransferParams,
)
from solana.rpc.api import Client
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.null_signer import NullSigner

load_dotenv()


PRIVY_SECRET = os.getenv("PRIVY_SECRET")
PRIVY_ID = os.getenv("PRIVY_ID")
PRIVY_WALLET_URL = os.getenv("PRIVY_WALLET_URL")
MAINNET_RPC_URL = os.getenv("MAINNET_RPC_URL")
PAYER_ADDRESS = os.getenv("PAYER_ADDRESS")
MINT_ADDRESS = os.getenv("USDC_MINT_ADDRESS")
WALLET_ID = os.getenv("WALLET_ID")

def _short(x: str | None, n: int = 6) -> str:
    if not x:
        return "None"
    x = str(x)
    return (x[:n] + "...") if len(x) > n else x

def privy_sign_and_send_sponsored(wallet_id: str, base64_tx: str):
    """
    Sends a base64-encoded Solana transaction to Privy for sign+send with sponsorship enabled.
    Logs status code and short error snippets; never logs secrets or full tx.
    """
    auth = base64.b64encode(f"{PRIVY_ID}:{PRIVY_SECRET}".encode()).decode()

    print("[pay] privy_send start", {
        "wallet_id": _short(wallet_id),
        "tx_b64_len": len(base64_tx or ""),
    })

    try:
        resp = requests.post(
            f"https://api.privy.io/v1/wallets/{wallet_id}/rpc",
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
                "privy-app-id": PRIVY_ID,
            },
            json={
                "method": "signAndSendTransaction",
                "caip2": "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",  # mainnet
                "sponsor": True,
                "params": {"transaction": base64_tx, "encoding": "base64"},
            },
            timeout=30,
        )
    except RequestException as e:
        print("[pay] ERR_PRIVY_HTTP", {"wallet_id": _short(wallet_id), "err": repr(e)})
        raise

    if resp.status_code >= 400:
        print("[pay] ERR_PRIVY_STATUS", {
            "wallet_id": _short(wallet_id),
            "status": resp.status_code,
            "body": resp.text[:250],
        })
    else:
        # don't print full body; just a small hint that it succeeded
        print("[pay] privy_send ok", {"wallet_id": _short(wallet_id), "status": resp.status_code})

    return resp

def build_create_ata_tx_base64(
    payer_address: str,   # pays fees (Privy will sign this)
    owner_address: str,   # who will own the ATA
    mint_address: str,    # token mint (e.g., USDC)
    rpc_url: str = MAINNET_RPC_URL,
) -> dict:
    """
    Builds a create-ATA transaction (base64) if ATA doesn't exist.
    """
    payer = Pubkey.from_string(payer_address)
    owner = Pubkey.from_string(owner_address)
    mint = Pubkey.from_string(mint_address)
    ata = get_associated_token_address(owner, mint)

    try:
        rpc = Client(rpc_url)
        if rpc.get_account_info(ata).value is not None:
            print("[pay] ata_create skip_exists", {"owner": _short(owner_address), "ata": _short(str(ata))})
            return {"ata": str(ata), "already_exists": True, "transaction": None}
    except Exception as e:
        print("[pay] ERR_ATA_EXISTS_CHECK", {"owner": _short(owner_address), "ata": _short(str(ata)), "err": repr(e)})
        raise

    try:
        ix = create_associated_token_account(payer=payer, owner=owner, mint=mint)
        recent_blockhash = rpc.get_latest_blockhash().value.blockhash
    except Exception as e:
        print("[pay] ERR_BLOCKHASH_OR_IX", {"payer": _short(payer_address), "owner": _short(owner_address), "err": repr(e)})
        raise

    try:
        msg = MessageV0.try_compile(
            payer=payer,
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=recent_blockhash,
        )
        tx = VersionedTransaction(msg, [NullSigner(payer)])  # Privy signs payer
        tx_b64 = base64.b64encode(bytes(tx)).decode("utf-8")
        print("[pay] ata_create built", {"owner": _short(owner_address), "ata": _short(str(ata)), "tx_b64_len": len(tx_b64)})
        return {"ata": str(ata), "already_exists": False, "transaction": tx_b64}
    except Exception as e:
        print("[pay] ERR_BUILD_ATA_TX", {"payer": _short(payer_address), "owner": _short(owner_address), "err": repr(e)})
        raise

async def create_privy_wallet(id: str) -> str:
    """
    Generate a new non-custodial Solana wallet using Privy if the user doesn't already have one.
    Stores wallet_id + address in Mongo.
    """
    short_id = _short(id)
    print("[pay] create_wallet start", {"id": short_id})

    # DB check (Motor async)
    try:
        record = await users().find_one({"id": id}, {"_id": 0})
    except Exception as e:
        print("[pay] ERR_DB_FIND_USER", {"id": short_id, "err": repr(e)})
        raise

    if record and record.get("address") and record.get("wallet_id"):
        print("[pay] create_wallet already_exists", {"id": short_id, "address": _short(record.get("address"))})
        return "User already has a wallet " + record["address"]

    auth_string = f"{PRIVY_ID}:{PRIVY_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()

    payload = {"chain_type": "solana"}
    headers = {
        "privy-app-id": PRIVY_ID,
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/json",
    }

    # Privy wallet create
    try:
        response = requests.post(PRIVY_WALLET_URL, json=payload, headers=headers, timeout=30)
    except RequestException as e:
        print("[pay] ERR_PRIVY_CREATE_HTTP", {"id": short_id, "err": repr(e)})
        raise

    if response.status_code >= 400:
        print("[pay] ERR_PRIVY_CREATE_STATUS", {"id": short_id, "status": response.status_code, "body": response.text[:250]})
        raise RuntimeError("Privy wallet creation failed")

    response_json = response.json()
    print("[pay] create_wallet privy_ok", {
        "id": short_id,
        "address": _short(response_json.get("address")),
        "wallet_id": _short(response_json.get("id")),
    })

    # Create USDC ATA for new wallet (sponsored via configured WALLET_ID payer)
    try:
        res = build_create_ata_tx_base64(
            payer_address=PAYER_ADDRESS,
            owner_address=response_json["address"],
            mint_address=MINT_ADDRESS,
        )
        print("[pay] create_wallet ata_status", {"already_exists": res["already_exists"], "ata": _short(res["ata"])})
    except Exception as e:
        print("[pay] ERR_CREATE_WALLET_ATA_BUILD", {"id": short_id, "err": repr(e)})
        raise

    if not res["already_exists"] and res["transaction"]:
        try:
            resp = privy_sign_and_send_sponsored(WALLET_ID, res["transaction"])
            print("[pay] create_wallet ata_sent", {"status": resp.status_code, "body": resp.text[:200]})
        except Exception as e:
            print("[pay] ERR_CREATE_WALLET_ATA_SEND", {"id": short_id, "err": repr(e)})
            raise

    # Insert DB record (Motor async)
    try:
        await users().insert_one(
            {
                "id": id,
                "address": response_json["address"],
                "wallet_id": response_json["id"],
            }
        )
        print("[pay] create_wallet db_insert_ok", {"id": short_id})
    except Exception as e:
        print("[pay] ERR_DB_INSERT_USER", {"id": short_id, "err": repr(e)})
        raise

    return response_json["address"]

'''
if __name__ == "__main__":
    import os
    import asyncio
    from db import connect_mongo, close_mongo

    async def _main():
        connect_mongo()
        try:
            user_id = os.getenv("USER_ID", "test-user-1")
            addr = await create_privy_wallet(user_id)
            print("Wallet address:", addr)
        finally:
            close_mongo()

    asyncio.run(_main())
'''
