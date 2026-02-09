import qrcode

solana_pay_url = "solana:4RqXDrEr8itNo5kQU8vERg2UBV4QrPYRV2D6DJxovp93?amount=0.00&spl-token=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&label=ChatPay&message=Cart%20checkout&reference=32ndXWEixp4KnzomcE6nikoK2uHb9UJkoL3bE3dv3Qux"

img = qrcode.make(solana_pay_url)
img.save("payment_qr.png")
