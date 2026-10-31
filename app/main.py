import importlib

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent import run_resolver, to_devanagari
from app.shield import run_shift
from app.shield import state as shield_state
from app.voice import speech_to_text, text_to_speech

app = FastAPI(title="Failed Payment Fixer")


class ChatIn(BaseModel):
    message: str
    session_id: str = "demo"
    speak: bool = True


async def _answer(message: str, session_id: str, speak: bool) -> dict:
    try:
        result = await run_in_threadpool(run_resolver, message, session_id)
    except httpx.RemoteProtocolError:  # Supabase dropped an idle connection; retry once
        result = await run_in_threadpool(run_resolver, message, session_id)
    result["audio"] = None
    if speak and result["show"]:
        result["say"] = await run_in_threadpool(to_devanagari, result["show"])
        try:
            result["audio"] = await text_to_speech(result["say"])
        except httpx.HTTPError:
            pass  # text still works if voice fails
    return result


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
async def chat(body: ChatIn):
    return await _answer(body.message, body.session_id, body.speak)


@app.post("/voice")
async def voice(file: UploadFile = File(...), session_id: str = Form("demo"),
                speak: bool = Form(True)):
    try:
        heard = await speech_to_text(await file.read())
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Speech to text failed: {e}")
    if not heard:
        raise HTTPException(422, "Didn't catch that. Please try again.")
    return {"heard": heard, **await _answer(heard, session_id, speak)}



# ---------- Shield teammate (ops) ----------
@app.get("/ops")
def ops():
    return FileResponse("static/ops.html")


@app.get("/shield/state")
def shield_state_api():
    return shield_state()


@app.post("/shield/run")
async def shield_run():
    try:
        result = await run_in_threadpool(run_shift)
    except httpx.RemoteProtocolError:  # Supabase dropped an idle connection; retry once
        result = await run_in_threadpool(run_shift)
    return {**result, **await run_in_threadpool(shield_state)}


@app.post("/shield/reset")
def shield_reset():
    """Reset demo data so the shift can be shown again."""
    import scripts.seed as seed
    importlib.reload(seed).main()  # reload so seed times are relative to now
    return shield_state()


from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def any_error(request: Request, exc: Exception):
    """Show the real error in the UI instead of a blank crash."""
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
