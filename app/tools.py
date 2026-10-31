"""The Resolver's five tools. Each one does one job and returns plain data."""
import math
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from app import config
from app.db import db
import contextvars

# Amounts the customer mentioned in the current message (set by the agent per request).
MENTIONED_AMOUNTS: contextvars.ContextVar[tuple] = contextvars.ContextVar("mentioned_amounts", default=())
LAST_MENTIONED = [()]  # backup channel for the same amounts


def _mismatch(t: dict) -> dict | None:
    """Block actions on a payment whose amount the customer never mentioned."""
    said = MENTIONED_AMOUNTS.get() or LAST_MENTIONED[0]
    amount = int(float(t["amount"]))
    if said and amount not in said:
        return {"error": f"Customer mentioned Rs {', '.join(map(str, said))}, but {t['id']} is Rs {amount}. "
                         "Do not act on it. Tell the customer no such payment was found and ask which one they mean."}
    return None

HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
             "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


def _when(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _deadline(txn: dict) -> datetime:
    return _when(txn["created_at"]) + timedelta(hours=config.DEADLINE_HOURS)


def _en(d: datetime) -> str:
    return f"{d.day} {d.strftime('%B')}"


def _hi(d: datetime) -> str:
    return f"{d.day} {HI_MONTHS[d.month - 1]}"


def _get(txn_id: str) -> dict:
    """Fetch one payment, and only if it belongs to the current customer."""
    rows = (db.table("transactions").select("*")
            .eq("id", txn_id.strip().upper()).eq("customer", config.CUSTOMER_ID)
            .limit(1).execute().data)
    if not rows:
        raise ValueError(f"No payment {txn_id} for this customer")
    return rows[0]


def _ticket(txn_id: str, kind: str, summary: str, amount: float = 0) -> str:
    row = db.table("tickets").insert({
        "txn_id": txn_id, "kind": kind, "summary": summary,
        "amount_inr": amount, "created_by": "resolver",
    }).execute().data[0]
    return f"PTM-{5000 + row['id']}"


STATUS_MEANING = {
    "success": "reached the payee",
    "pending": "money debited but not yet credited to the payee",
    "refund_pending": "payment failed, refund not yet returned",
    "reversed": "already refunded to the customer (with compensation if it was late)",
    "repushed": "re-pushed by Shield, the money has gone through",
}


@tool
def find_transactions(amount: float | None = None, payee: str | None = None) -> dict:
    """Find the customer's recent UPI payments, newest first.
    Pass amount (in rupees) if the customer mentioned one, and/or payee if they named
    a person or shop (for example "Priya" or "Swiggy"). If nothing matches,
    the most recent payments are returned instead."""
    def fetch(amt=None, name=None):
        q = (db.table("transactions").select("id,amount,payee,status,created_at")
             .eq("customer", config.CUSTOMER_ID).order("created_at", desc=True).limit(10))
        if amt:
            q = q.eq("amount", amt)
        if name:
            q = q.ilike("payee", f"%{name.strip()}%")
        return [{"txn_id": r["id"], "amount": r["amount"], "payee": r["payee"], "status": r["status"],
                 "status_meaning": STATUS_MEANING.get(r["status"], r["status"]), "time": r["created_at"]}
                for r in q.execute().data]

    payments = fetch(amount, payee)
    if payments:
        return {"matched": True, "payments": payments}
    asked = " and ".join(x for x in [f"Rs {amount}" if amount else "", payee or ""] if x)
    return {"matched": False,
            "note": (f"No payment matching {asked}. Do NOT assume it is one of the payments below. "
                     "Tell the customer it was not found and ask which payment they mean."),
            "recent_payments": fetch()}


@tool
def check_status(txn_id: str) -> dict:
    """Get the real status of one payment: did it reach the payee, where is it
    stuck, and the RBI T+1 deadline (deadline_date in English, deadline_hindi)."""
    t = _get(txn_id)
    dl = _deadline(t)
    return {
        "txn_id": t["id"], "amount": t["amount"], "payee": t["payee"], "bank": t["bank"],
        "status": t["status"], "stage": t["stage"],
        "reached_payee": t["status"] == "success",
        "deadline_date": _en(dl), "deadline_hindi": _hi(dl),
        "deadline_passed": datetime.now(timezone.utc) > dl,
    }


@tool
def raise_dispute(txn_id: str) -> dict:
    """Raise a dispute with the bank for a stuck payment that is still within
    its deadline. Returns the ticket number and the auto-refund date."""
    t = _get(txn_id)
    if (blocked := _mismatch(t)):
        return blocked
    dl = _deadline(t)
    ticket = _ticket(t["id"], "dispute", f"Stuck at: {t['stage']}")
    return {"ticket": ticket, "auto_refund_by": _en(dl), "auto_refund_by_hindi": _hi(dl)}


@tool
def calc_compensation(txn_id: str) -> dict:
    """For a payment past its deadline: days late and compensation owed
    (RBI TAT rule for UPI: Rs 100 per day after T+1)."""
    t = _get(txn_id)
    late = datetime.now(timezone.utc) - _deadline(t)
    days = max(0, math.ceil(late.total_seconds() / 86400))
    return {"txn_id": t["id"], "days_late": days, "compensation_inr": days * 100}


@tool
def escalate_to_human(txn_id: str, summary: str) -> dict:
    """Send a late or unusual case to a human agent, with a short summary.
    Returns the ticket number."""
    t = _get(txn_id)
    if (blocked := _mismatch(t)):
        return blocked
    return {"ticket": _ticket(t["id"], "escalation", summary[:300])}


TOOLS = [find_transactions, check_status, raise_dispute, calc_compensation, escalate_to_human]
