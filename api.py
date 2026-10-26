import logging
import os
import time

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from pidetect.classifier import get_classifier

log = logging.getLogger("pidetect")
MAX_CHARS = int(os.getenv("MAX_CHARS", "8000"))

app = FastAPI(title="Prompt Injection Detector")


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)


class GuardRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_CHARS)
    payload: dict = Field(default_factory=dict)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify(req: ClassifyRequest, clf=Depends(get_classifier)):
    start = time.perf_counter()
    r = clf.predict(req.text)
    ms = round((time.perf_counter() - start) * 1000, 1)
    log.info("classify label=%s score=%.3f ms=%s", r.label, r.score, ms)
    return {"label": r.label, "score": r.score, "is_injection": r.is_injection, "latency_ms": ms}


@app.post("/classify/batch")
def classify_batch(req: BatchRequest, clf=Depends(get_classifier)):
    if any(len(t) > MAX_CHARS for t in req.texts):
        raise HTTPException(422, f"each text must be under {MAX_CHARS} characters")
    return {"results": [r.__dict__ for r in clf.predict_batch(req.texts)]}


@app.post("/guard")
async def guard(req: GuardRequest, clf=Depends(get_classifier)):
    r = clf.predict(req.prompt)
    if r.is_injection:
        log.warning("blocked prompt score=%.3f", r.score)
        raise HTTPException(403, {"error": "prompt_injection_detected", "score": r.score})

    upstream = os.getenv("UPSTREAM_URL")
    if not upstream:
        return {"allowed": True, "score": r.score}

    headers = {}
    if os.getenv("UPSTREAM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['UPSTREAM_API_KEY']}"
    body = {**req.payload, "prompt": req.prompt}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(upstream, json=body, headers=headers)
        except httpx.HTTPError:
            raise HTTPException(502, "upstream unavailable")
    return {"allowed": True, "score": r.score, "upstream_status": resp.status_code, "response": resp.json()}
