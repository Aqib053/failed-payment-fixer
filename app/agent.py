"""Resolver teammate: a LangGraph agent that loops model -> tools -> model."""
import json
import re
import time
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app import config
from app.tools import TOOLS

SYSTEM_PROMPT = """You are Paytm's customer support teammate for failed or stuck UPI payments.
Work like this: find the payment (use the amount if they said one), check its status, then act.
- Reached the payee: just tell them it reached.
- Not reached and deadline not passed: raise_dispute.
- Deadline passed: calc_compensation, then escalate_to_human.
Never invent payments, dates or ticket numbers. Use only what the tools return.
When done with tools, reply with ONLY a JSON object: {"show": "...", "say": "..."}
"show": simple Hindi in Roman English letters (Hinglish), max 2 short sentences,
with the amount, date and ticket number. Example:
"Aapke ₹2000 31 October tak wapas aa jayenge. Shikayat PTM-5001 darj ho gayi hai."
"say": the same message in Devanagari script, for the Hindi voice."""


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = init_chat_model(config.LLM_MODEL, temperature=0).bind_tools(TOOLS)


def call_model(state: State) -> dict:
    return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}


graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # tool call? -> tools, else end
graph.add_edge("tools", "agent")
resolver = graph.compile(checkpointer=MemorySaver())  # remembers each session


def _text(content) -> str:
    """Some models return a list of parts instead of one string."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _parse(raw: str) -> dict:
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0) if match else raw)
        show = str(data.get("show", "")).strip()
        return {"show": show, "say": str(data.get("say") or show).strip()}
    except (json.JSONDecodeError, AttributeError):
        return {"show": raw, "say": raw}


def run_resolver(message: str, session_id: str) -> dict:
    cfg = {"configurable": {"thread_id": session_id}}
    before = len(resolver.get_state(cfg).values.get("messages", []))
    start = time.perf_counter()
    result = resolver.invoke({"messages": [HumanMessage(message)]}, cfg)
    new = result["messages"][before:]
    steps = [{"tool": c["name"], "args": c["args"]}
             for m in new if isinstance(m, AIMessage) for c in m.tool_calls]
    reply = _parse(_text(new[-1].content))
    return {**reply, "steps": steps, "seconds": round(time.perf_counter() - start, 1)}
