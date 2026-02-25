import os
import stripe

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def create_stripe_checkout(amount: float, currency: str, metadata: dict):
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": currency.lower(),
                "product_data": {
                    "name": "Order Payment",
                },
                "unit_amount": int(amount * 100),  # cents
            },
            "quantity": 1,
        }],
        metadata=metadata,
        success_url="https://example.com/success",
        cancel_url="https://example.com/cancel",
    )

    return {
        "session_id": session.id,
        "checkout_url": session.url,
    }