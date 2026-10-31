"""Shield teammate: an autonomous ops agent that runs a "shift" over all stuck payments.
It decides per payment: re-push before the RBI deadline, pay compensation if late, tell the
customer, and escalate to a human. Tools enforce the rules, so it can't act wrongly."""
import json
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app import config
from app.agent import backups, primary
from app.db import db

STUCK = ["pending", "refund_pending"]
URGENT_HOURS = 12        # re-push when this close to the deadline
PENALTY_PER_DAY = 100    # RBI TAT: Rs 100 per day after T+1 (UPI)


def _hours_left(row: dict) -> float:
    created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
    deadline = created + timedelta(hours=config.DEADLINE_HOURS)
    return round((deadline - datetime.now(timezone.utc)).total_seconds() / 3600, 1)


def _row(txn_id: str) -> dict | None:
    rows = db.table("transactions").select("*").eq("id", txn_id.strip().upper()).limit(1).execute().data
    return rows[0] if rows else None


def _log(txn_id: str, kind: str, summary: str, amount: float = 0) -> str:
    row = db.table("tickets").insert({"txn_id": txn_id, "kind": kind, "summary": summary,
                                      "amount_inr": amount, "created_by": "shield"}).execute().data[0]
    return f"PTM-{5000 + row['id']}"


def _set(txn_id: str, status: str, stage: str) -> None:
    db.table("transactions").update({"status": status, "stage": stage}).eq("id", txn_id).execute()


@tool
def list_stuck_payments() -> dict:
    """List every stuck UPI payment (all customers) with hours left before the RBI deadline.
    hours_left <= 0 means the deadline has already passed (late)."""
    rows = db.table("transactions").select("*").in_("status", STUCK).execute().data
    items = [{"txn_id": r["id"], "customer": r["customer"], "amount": int(float(r["amount"])),
              "stage": r["stage"], "hours_left": _hours_left(r)} for r in rows]
    items.sort(key=lambda x: x["hours_left"])
    return {"count": len(items), "payments": items}


@tool
def repush_payment(txn_id: str) -> dict:
    """Re-push a stuck payment to the bank so it completes before the deadline.
    Only for payments that are NOT late and have 12 hours or less left."""
    r = _row(txn_id)
    if not r or r["status"] not in STUCK:
        return {"error": f"{txn_id} is not a stuck payment."}
    left = _hours_left(r)
    if left <= 0:
        return {"error": f"{txn_id} is already late. Use pay_compensation instead."}
    if left > URGENT_HOURS:
        return {"error": f"{txn_id} has {left}h left. Not urgent yet, keep watching."}
    _set(r["id"], "repushed", "Re-pushed by Shield before the deadline")
    ticket = _log(r["id"], "repush", f"Re-pushed with {left}h left", PENALTY_PER_DAY)
    return {"ticket": ticket, "hours_left": left, "penalty_avoided_inr": PENALTY_PER_DAY}


@tool
def pay_compensation(txn_id: str) -> dict:
    """Refund a payment that missed the RBI deadline and pay Rs 100 per late day."""
    r = _row(txn_id)
    if not r or r["status"] not in STUCK:
        return {"error": f"{txn_id} is not a stuck payment."}
    left = _hours_left(r)
    if left > 0:
        return {"error": f"{txn_id} is not late yet ({left}h left). Use repush_payment if urgent."}
    days = max(1, math.ceil(-left / 24))
    amount = days * PENALTY_PER_DAY
    _set(r["id"], "reversed", f"Refunded + Rs {amount} compensation")
    ticket = _log(r["id"], "compensation", f"{days} day(s) late", amount)
    return {"ticket": ticket, "days_late": days, "compensation_inr": amount}


@tool
def notify_customer(txn_id: str, message: str) -> dict:
    """Send the customer a short update, in Hindi written with English letters."""
    r = _row(txn_id)
    if not r:
        return {"error": f"{txn_id} not found."}
    _log(r["id"], "notification", message[:300])
    return {"sent_to": r["customer"]}


@tool
def escalate(txn_id: str, summary: str) -> dict:
    """Hand a case to a human agent with a one-line summary."""
    r = _row(txn_id)
    if not r:
        return {"error": f"{txn_id} not found."}
    return {"ticket": _log(r["id"], "escalation", summary[:300])}


@tool
def finish_shift(report: str) -> str:
    """End the shift. report: 2 short lines in English for the human ops lead."""
    return "done"


TOOLS = [list_stuck_payments, repush_payment, pay_compensation, notify_customer, escalate, finish_shift]

PROMPT = f"""You are Shield, Paytm's autonomous operations teammate. You run a shift over all
stuck UPI payments, with no human involved.
1. Call list_stuck_payments once.
2. For every payment in the list:
   - Late (hours_left <= 0): pay_compensation, then notify_customer, then escalate with a one-line summary.
   - hours_left <= {URGENT_HOURS}: repush_payment, then notify_customer.
   - Otherwise: leave it, it is still safe.
   notify_customer message: short Hindi in English letters, e.g.
   "Aapke Rs 3500 wapas aa gaye hain. Deri ke liye maafi." Never use special symbols.
You may call several tools at once. Never act on a payment that is not in the list.
When every payment is handled, call finish_shift with a 2-line report for the ops lead."""


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = primary.bind_tools(TOOLS).with_fallbacks([b.bind_tools(TOOLS) for b in backups])


def _think(state: State) -> dict:
    return {"messages": [llm.invoke([SystemMessage(PROMPT), *state["messages"]])]}


def _after_tools(state: State) -> str:
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    return END if any(c["name"] == "finish_shift" for c in last_ai.tool_calls) else "agent"


_g = StateGraph(State)
_g.add_node("agent", _think)
_g.add_node("tools", ToolNode(TOOLS))
_g.add_edge(START, "agent")
_g.add_conditional_edges("agent", tools_condition)
_g.add_conditional_edges("tools", _after_tools)
shield = _g.compile()


def _load(content) -> dict:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return {"text": str(content)}


def run_shift() -> dict:
    start = time.perf_counter()
    result = shield.invoke({"messages": [HumanMessage("Start your shift now.")]},
                           {"recursion_limit": 80})
    outputs = {m.tool_call_id: _load(m.content) for m in result["messages"] if isinstance(m, ToolMessage)}
    actions, report = [], ""
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                if c["name"] == "finish_shift":
                    report = c["args"].get("report", "")
                else:
                    actions.append({"tool": c["name"], "args": c["args"], "result": outputs.get(c["id"], {})})
    return {"actions": actions, "report": report, "seconds": round(time.perf_counter() - start, 1)}


def state() -> dict:
    """Everything the ops dashboard shows."""
    txns = db.table("transactions").select("*").neq("status", "success").execute().data
    payments = [{"txn_id": r["id"], "customer": r["customer"], "amount": int(float(r["amount"])),
                 "stage": r["stage"], "status": r["status"], "hours_left": _hours_left(r)} for r in txns]
    payments.sort(key=lambda p: (p["status"] not in STUCK, p["hours_left"]))
    tickets = db.table("tickets").select("kind,amount_inr").eq("created_by", "shield").execute().data
    count = lambda k: sum(1 for t in tickets if t["kind"] == k)
    return {"payments": payments, "stats": {
        "fixed_without_human": count("repush") + count("compensation"),
        "penalty_avoided": count("repush") * PENALTY_PER_DAY,
        "compensation_paid": int(sum(float(t["amount_inr"] or 0) for t in tickets if t["kind"] == "compensation")),
        "customers_notified": count("notification"),
        "escalated": count("escalation"),
    }}
