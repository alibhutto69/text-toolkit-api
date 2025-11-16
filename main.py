from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="Text Toolkit API", version="1.0.0")

class AnalyzeRequest(BaseModel):
    text: str
    max_summary_words: int = 80

class AnalyzeResponse(BaseModel):
    summary: str
    keywords: list[str]
    sentiment: str

async def call_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Calls local Ollama server with a prompt and returns the response text.
    Make sure `ollama serve` is running (it usually runs automatically).
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Ollama error")
    data = resp.json()
    return data.get("response", "").strip()

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    # 1) Summary
    summary_prompt = (
        f"Summarize the following text in no more than {req.max_summary_words} words. "
        "Use plain English.\n\n"
        f"TEXT:\n{req.text}"
    )
    summary = await call_ollama(summary_prompt)

    # 2) Keywords (return as JSON list)
    keywords_prompt = (
        "Extract 5-10 important keywords or key phrases from the text below. "
        "Return ONLY a JSON array of strings, nothing else.\n\n"
        f"TEXT:\n{req.text}"
    )
    keywords_raw = await call_ollama(keywords_prompt)
    try:
        keywords = json.loads(keywords_raw)
        if not isinstance(keywords, list):
            raise ValueError
        keywords = [str(k) for k in keywords]
    except Exception:
        # fallback: split by comma
        keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    # 3) Sentiment
    sentiment_prompt = (
        "Decide whether the sentiment of the following text is positive, "
        "neutral, or negative. Answer with ONE WORD: positive, neutral, or negative.\n\n"
        f"TEXT:\n{req.text}"
    )
    sentiment = (await call_ollama(sentiment_prompt)).lower()
    if "positive" in sentiment:
        sentiment = "positive"
    elif "negative" in sentiment:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return AnalyzeResponse(
        summary=summary,
        keywords=keywords,
        sentiment=sentiment,
    )
