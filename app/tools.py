"""The Resolver's five tools. Each one does one job and returns plain data."""
import math
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from app import config
from app.db import db

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


@tool
def find_transactions(amount: float | None = None) -> list[dict]:
    """Find the customer's recent UPI payments, newest first.
    Pass amount (in rupees) if the customer mentioned one."""
    q = (db.table("transactions").select("id,amount,payee,status,created_at")
         .eq("customer", config.CUSTOMER_ID).order("created_at", desc=True).limit(10))
    if amount:
        q = q.eq("amount", amount)
    return [{"txn_id": r["id"], "amount": r["amount"], "payee": r["payee"],
             "status": r["status"], "time": r["created_at"]} for r in q.execute().data]


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
    return {"ticket": _ticket(t["id"], "escalation", summary[:300])}


TOOLS = [find_transactions, check_status, raise_dispute, calc_compensation, escalate_to_human]
