# Failed Payment Fixer

An autonomous AI teammate that fixes stuck UPI payments by Hindi voice.
Built for the Paytm Build for India AI Hackathon (Track 3).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in your keys. Never commit .env
```

1. In Supabase SQL Editor, run `db/schema.sql`.
2. Seed demo data: `python -m scripts.seed`
3. Start: `uvicorn app.main:app --reload`
4. Open http://localhost:8000 in Chrome.

## How it works

Voice (Sarvam Saaras) -> LangGraph Resolver agent -> 5 tools on Supabase
-> reply as `show` (Roman Hindi) + `say` (Devanagari) -> voice (Sarvam Bulbul).
