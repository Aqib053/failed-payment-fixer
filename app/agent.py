"""Resolver teammate: a LangGraph agent that loops model -> tools -> model,
and finishes by calling the `reply` tool with the final answer."""
import json
import re
import time
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app import config
from app import tools as _tools
from app.tools import MENTIONED_AMOUNTS, TOOLS, find_transactions


@tool
def reply(show: str) -> str:
    """Send the final answer to the customer. Call this exactly once, alone, as the last step.
    show: simple Hindi written ONLY in English letters (Hinglish), max 2 short sentences.
    Write money as Rs 2000 (no special symbols)."""
    return "sent"


SYSTEM_PROMPT = """You are Paytm's customer support teammate for failed or stuck UPI payments.
Work like this: find the payment (use the amount if they said one), check its status, then act.
- Reached the payee: just tell them it reached.
- Not reached and deadline not passed: raise_dispute.
- Deadline passed: calc_compensation, then escalate_to_human.
Never invent payments, dates or ticket numbers. Use only what the tools return.
Finish by calling the `reply` tool once, with the amount, date and ticket number in it.
Example show: "Aapke Rs 2000 31 October tak wapas aa jayenge. Shikayat PTM-5001 darj ho gayi hai." """

ALL_TOOLS = [*TOOLS, reply]


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _model(name: str):
    """Reasoning models (gpt-5, o-series) only accept the default temperature."""
    fixed = any(k in name for k in ("gpt-5", ":o1", ":o3", ":o4"))
    return init_chat_model(name) if fixed else init_chat_model(name, temperature=0)


primary = _model(config.LLM_MODEL)
backups = [_model(config.LLM_FALLBACK_MODEL)] if config.LLM_FALLBACK_MODEL else []
llm = primary.bind_tools(ALL_TOOLS).with_fallbacks([b.bind_tools(ALL_TOOLS) for b in backups])
base_llm = primary.with_fallbacks(backups)


def call_model(state: State) -> dict:
    return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}


def after_tools(state: State) -> str:
    """Stop once the agent has sent its reply; otherwise let it think again."""
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    return END if any(c["name"] == "reply" for c in last_ai.tool_calls) else "agent"


graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # tool call? -> tools, else end
graph.add_conditional_edges("tools", after_tools)
resolver = graph.compile(checkpointer=MemorySaver())  # remembers each session


def _text(content) -> str:
    """Some models return a list of parts instead of one string."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _rupee(text: str) -> str:
    """Turn 'Rs 2000' / 'Rs. 2000' / 'INR 2000' into '₹2000' for display."""
    return re.sub(r"\b(?:Rs\.?|INR)\s?(?=\d)", "₹", text)


def to_devanagari(text: str) -> str:
    """Rewrite a Hinglish reply in Devanagari so the Hindi voice reads it naturally.
    A plain text call (no tools, no JSON), so Hindi script can't break parsing."""
    prompt = ("Rewrite this Hindi sentence, which is written in English letters, in Devanagari script. "
              "Keep numbers, ₹ amounts, dates, ticket IDs and brand names exactly as they are. "
              "Output only the rewritten sentence.\n\n" + text)
    try:
        out = _text(base_llm.invoke(prompt).content).strip()
        return out or text
    except Exception:
        return text  # voice still works, just less natural


def _parse(raw: str) -> dict:
    """Fallback if a model answers in plain text instead of calling `reply`."""
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0) if match else raw)
        show = str(data.get("show", "")).strip()
        return {"show": show, "say": str(data.get("say") or show).strip()}
    except (json.JSONDecodeError, AttributeError):
        return {"show": raw, "say": raw}


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


import re
import unicodedata

def _norm(s: str) -> str:
    """Lowercase, drop the nukta dot, so हज़ार and हजार match the same key."""
    return unicodedata.normalize("NFD", s.lower()).replace("\u093c", "")

_UNITS = {_norm(k): v for k, v in {
    "ek": 1, "एक": 1, "do": 2, "दो": 2, "teen": 3, "tin": 3, "तीन": 3, "char": 4, "chaar": 4, "चार": 4,
    "paanch": 5, "panch": 5, "pach": 5, "पांच": 5, "पाँच": 5, "chhe": 6, "chhah": 6, "che": 6, "छह": 6, "छः": 6, "छे": 6,
    "saat": 7, "सात": 7, "aath": 8, "aat": 8, "आठ": 8, "nau": 9, "नौ": 9, "das": 10, "dus": 10, "दस": 10,
    "gyarah": 11, "gyara": 11, "ग्यारह": 11, "barah": 12, "bara": 12, "baarah": 12, "बारह": 12,
    "terah": 13, "तेरह": 13, "chaudah": 14, "chauda": 14, "चौदह": 14, "pandrah": 15, "pandra": 15, "पंद्रह": 15,
    "solah": 16, "sola": 16, "सोलह": 16, "satrah": 17, "सत्रह": 17, "atharah": 18, "athara": 18, "अठारह": 18,
    "unnis": 19, "unnees": 19, "उन्नीस": 19, "bees": 20, "bis": 20, "बीस": 20, "pachchis": 25, "pachees": 25, "पच्चीस": 25,
    "tees": 30, "tis": 30, "तीस": 30, "chalis": 40, "chalees": 40, "चालीस": 40, "pachas": 50, "pachaas": 50, "पचास": 50,
    "saath": 60, "साठ": 60, "sattar": 70, "सत्तर": 70, "assi": 80, "अस्सी": 80, "nabbe": 90, "नब्बे": 90,
}.items()}
_HALVES = {_norm(k): v for k, v in {"dedh": 1.5, "derh": 1.5, "डेढ़": 1.5, "डेढ": 1.5,
                                     "dhai": 2.5, "dhaai": 2.5, "ढाई": 2.5}.items()}
_PLUS_HALF = {_norm(k) for k in ["sadhe", "saadhe", "saade", "साढ़े", "साढे"]}
_MULT = {_norm(k): v for k, v in {
    "sau": 100, "सौ": 100, "hundred": 100,
    "hazaar": 1000, "hazar": 1000, "hajar": 1000, "hajaar": 1000, "हज़ार": 1000, "हजार": 1000, "thousand": 1000,
    "lakh": 100000, "lac": 100000, "लाख": 100000,
}.items()}
_RUPEE = {_norm(k) for k in ["rupaye", "rupaya", "rupay", "rupees", "rupee", "rs", "inr", "₹",
                             "रुपये", "रुपए", "रुपया", "रूपये", "रूपए"]}
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def amounts_in(text: str) -> tuple:
    """Rupee amounts in what the customer said, in digits or Hindi/Hinglish words.
    '₹100' -> (100,), 'do hazaar' -> (2000,), 'बारह सौ रुपये' -> (1200,), 'sau rupaye' -> (100,)."""
    t = _norm(text.translate(_DIGITS)).replace(",", "")
    t = re.sub(r"(₹|rs\.?)\s*(?=\d)", " ", t)          # '₹2000' / 'rs2000' -> ' 2000'
    toks = [w.strip(".:;'\"()") for w in re.split(r"[\s?!।|/]+", t)]
    toks = [w for w in toks if w]
    found, total, cur, active, plus_half = [], 0.0, 0.0, False, False

    def flush():
        nonlocal total, cur, active, plus_half
        if active:
            amount = int(round(total + cur))
            if 10 <= amount <= 10_000_000:
                found.append(amount)
        total, cur, active, plus_half = 0.0, 0.0, False, False

    for i, w in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        leads = nxt in _MULT or nxt in _RUPEE           # a number word only counts before sau/hazaar/rupaye
        if w.replace(".", "", 1).isdigit():
            n = float(w)
            if nxt in _MULT:
                cur, active = n, True
            else:
                flush(); cur, active = n, True; flush()
        elif w in _PLUS_HALF and (nxt in _UNITS):
            plus_half = True
        elif w in _UNITS and (leads or active):
            v = _UNITS[w] + (0.5 if plus_half else 0)
            plus_half = False
            if active and nxt not in _MULT:
                cur += v                                  # 'chaar sau PACHAS' -> 450
            else:
                total, cur, active = total + (cur if active else 0), v, True
        elif w in _HALVES and nxt in _MULT:
            cur, active = _HALVES[w], True
        elif w in _MULT:
            m = _MULT[w]
            if m == 100:
                cur = (cur or 1) * 100
            else:
                total += (cur or 1) * m
                cur = 0.0
            active = True
        elif w in _RUPEE:
            flush()
        else:
            flush()
    flush()
    return tuple(dict.fromkeys(found))



def _not_found_reply(said: tuple) -> dict | None:
    """If the customer named an amount that matches none of their payments, answer directly."""
    data = find_transactions.invoke({})
    rows = data.get("payments") or data.get("recent_payments") or []
    if any(int(float(r["amount"])) in said for r in rows):
        return None
    recent = ", ".join(f"Rs {int(float(r['amount']))} {r['payee']}" for r in rows[:2])
    show = f"Mujhe Rs {said[0]} ka koi payment nahi mila."
    if recent:
        show += f" Aapke recent payments: {recent}. Kaunsa payment check karun?"
    return {"show": _rupee(show), "say": _rupee(show),
            "steps": [{"tool": "find_transactions", "args": {"amount": said[0]}}]}


def run_resolver(message: str, session_id: str) -> dict:
    cfg = {"configurable": {"thread_id": session_id}}
    before = len(resolver.get_state(cfg).values.get("messages", []))
    start = time.perf_counter()
    said = amounts_in(message)
    hint = (f"\n\n[Note: customer mentioned amount(s): {', '.join(map(str, said))}. "
            "Pass it to find_transactions and only act on a payment with this amount.]") if said else ""
    if said:
        early = _not_found_reply(said)
        if early:
            resolver.update_state(cfg, {"messages": [HumanMessage(message), AIMessage(early["show"])]})
            return {**early, "seconds": round(time.perf_counter() - start, 1)}
    _tools.LAST_MENTIONED[0] = said
    token = MENTIONED_AMOUNTS.set(said)
    try:
        result = resolver.invoke({"messages": [HumanMessage(message + hint)]}, cfg)
    finally:
        MENTIONED_AMOUNTS.reset(token)
    new = result["messages"][before:]
    calls = [c for m in new if isinstance(m, AIMessage) for c in m.tool_calls]
    steps = [{"tool": c["name"], "args": c["args"]} for c in calls if c["name"] != "reply"]
    final = next((c["args"] for c in reversed(calls) if c["name"] == "reply"), None)
    if final:
        show = _rupee(str(final.get("show", "")).strip())
        answer = {"show": show, "say": show}
    else:
        answer = _parse(_text(new[-1].content))
        answer["show"] = _rupee(answer["show"])
    return {**answer, "steps": steps, "seconds": round(time.perf_counter() - start, 1)}
