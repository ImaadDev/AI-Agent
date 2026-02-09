# paylink.py
import urllib.parse as up
from typing import Optional

USDC_MINT_MAINNET = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def solana_pay_url(
    recipient: str,
    amount: str,
    *,
    spl_token: Optional[str] = USDC_MINT_MAINNET,  # set None for SOL
    label: str = "ChatPay",
    message: Optional[str] = None,
    memo: Optional[str] = None,
    reference: Optional[str] = None,
) -> str:
    """
    Build a Solana Pay URL.

    Example:
      solana:<recipient>?amount=5.25&spl-token=<USDC_MINT>&label=ChatPay&message=Invoice+%23123&memo=inv_123

    Args:
      recipient: Solana address (base58 string)
      amount: amount as a string (e.g., "5.25")
      spl_token: SPL token mint (USDC by default). Set to None to request SOL.
      label/message/memo/reference: optional Solana Pay params

    Returns:
      A solana: URL string
    """
    if not recipient or not isinstance(recipient, str):
        raise ValueError("recipient must be a non-empty string")
    if not amount or not isinstance(amount, str):
        raise ValueError("amount must be a non-empty string (e.g., '5.25')")

    params = {"amount": amount}

    if spl_token:
        params["spl-token"] = spl_token
    if label:
        params["label"] = label
    if message:
        params["message"] = message
    if memo:
        params["memo"] = memo
    if reference:
        params["reference"] = reference

    return f"solana:{recipient}?{up.urlencode(params, quote_via=up.quote)}"

'''
if __name__ == "__main__":
    # quick manual test
    recipient = "4RqXDrEr8itNo5kQU8vERg2UBV4QrPYRV2D6DJxovp93"
    url = solana_pay_url(
        recipient,
        "5.25",
        message="Invoice #123",
        memo="inv_123",
        # reference="A_UNIQUE_PUBKEY",
    )
    print(url)
'''
