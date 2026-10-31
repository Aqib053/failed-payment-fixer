"""Reset demo data. Times are relative to now, so deadlines always look real.

Run: python -m scripts.seed
"""
from datetime import datetime, timedelta, timezone

from app.db import db

now = datetime.now(timezone.utc)


def ago(hours: float) -> str:
    return (now - timedelta(hours=hours)).isoformat()


ROWS = [
    # The demo customer (Rahul)
    ("TXN84721", "rahul", 2000, "Sharma Kirana Store", "sharmakirana@ptys", "SBI", "pending",
     "Receiver's bank (SBI) did not confirm credit", 5),
    ("TXN83110", "rahul", 1200, "Priya Nair", "priya.n@okaxis", "Axis Bank", "refund_pending",
     "Payment failed, refund not yet returned", 50),
    ("TXN82990", "rahul", 450, "Swiggy", "swiggy@ptybl", "HDFC Bank", "success", "Delivered", 30),
    ("TXN82654", "rahul", 299, "Jio Recharge", "jio@ptybl", "Paytm Payments Bank", "success", "Delivered", 72),
    # Other customers: what the Shield teammate watches
    ("TXN80412", "meena", 3500, "Iyer Medicals", "iyermed@okicici", "ICICI", "pending", "Receiver bank timeout (ICICI)", 21),
    ("TXN80377", "arjun", 780, "Rao Stores", "raostores@ybl", "SBI", "pending", "NPCI response pending", 19),
    ("TXN80266", "farhan", 12000, "Khan Electronics", "khanelec@okhdfc", "PNB", "pending", "Receiver bank timeout (PNB)", 16),
    ("TXN80198", "kavya", 640, "Shetty Bakery", "shettybake@paytm", "Kotak", "refund_pending", "Refund not returned (Kotak)", 28),
    ("TXN80155", "rohan", 1500, "Das Tailors", "dastailor@ybl", "Bank of Baroda", "pending", "Receiver bank timeout (BoB)", 9),
    ("TXN80102", "sunita", 220, "Verma Dairy", "vermadairy@okaxis", "Canara", "refund_pending", "Refund not returned (Canara)", 40),
    ("TXN80044", "imran", 5000, "Sheikh Travels", "sheikhtrv@paytm", "SBI", "pending", "NPCI response pending", 3),
]


def main() -> None:
    db.table("tickets").delete().neq("id", 0).execute()
    db.table("transactions").delete().neq("id", "").execute()
    db.table("transactions").insert([
        {"id": i, "customer": c, "amount": a, "payee": p, "payee_vpa": v, "bank": b,
         "status": s, "stage": st, "created_at": ago(h)}
        for i, c, a, p, v, b, s, st, h in ROWS
    ]).execute()
    print(f"Seeded {len(ROWS)} transactions.")


if __name__ == "__main__":
    main()
