"""
FastAPI backend for the SPO Relationship Extractor.
Run with: uvicorn api:app --reload
"""

import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from relationship_extractor import RelationshipExtractor

load_dotenv(find_dotenv(), override=True)
_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

extractor: RelationshipExtractor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor
    if not _API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    extractor = RelationshipExtractor(api_key=_API_KEY, model=_MODEL)
    yield


app = FastAPI(title="SPO Relationship Extractor API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    text: str
    include_implicit: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_response(df: pd.DataFrame) -> dict:
    relationships = df.to_dict(orient="records")
    stats: dict = {}
    if not df.empty:
        stats = {
            "unique_subjects": int(df["subject"].nunique()),
            "unique_predicates": int(df["predicate"].nunique()),
            "unique_objects": int(df["object"].nunique()),
            "negated_count": int(df["negated"].sum()) if "negated" in df.columns else 0,
        }
    return {"relationships": relationships, "count": len(relationships), "stats": stats}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _MODEL}


@app.post("/extract")
async def extract(req: ExtractRequest):
    """Extract relationships from a plain-text body."""
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    try:
        df = extractor.extract(req.text, include_implicit=req.include_implicit)
        return _df_to_response(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/extract/file")
async def extract_file(
    file: Annotated[UploadFile, File()],
    include_implicit: Annotated[bool, Form()] = False,
):
    """Extract relationships from an uploaded .txt / .md / .pdf file."""
    filename = file.filename or ""
    raw = await file.read()

    if filename.lower().endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except ImportError as exc:
            raise HTTPException(
                status_code=422, detail="PyPDF2 not installed; cannot process PDF"
            ) from exc
    else:
        text = raw.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    try:
        df = extractor.extract(text, include_implicit=include_implicit)
        return _df_to_response(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
