"""
api.py
========
FastAPI backend for the TU CSIT college recommendation system (FYP).

Wraps recommendation_engine.py as a JSON API: loads college_master_ranking.csv,
predicted_cutoffs_2083.csv, and admission_classifier.pkl once at startup, then
serves college dropdown data and per-student recommendations over HTTP.

Run:
    uvicorn api:app --reload
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommendation_engine import (
    load_master_ranking,
    load_classifier,
    attach_predicted_cutoffs,
    recommend_from_preference_list,
    top5_by_cutoff,
    attach_admission_probability,
)

BASE_DIR = Path(__file__).resolve().parent
RANKING_PATH = BASE_DIR / "college_master_ranking.csv"
PREDICTED_PATH = BASE_DIR / "predicted_cutoffs_2083.csv"
CLASSIFIER_PATH = BASE_DIR / "admission_classifier.pkl"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not RANKING_PATH.exists():
        raise RuntimeError(
            f"{RANKING_PATH} not found — run build_master_dataset.py first."
        )
    master_df = load_master_ranking(str(RANKING_PATH))

    if PREDICTED_PATH.exists():
        master_df = attach_predicted_cutoffs(master_df, str(PREDICTED_PATH))

    classifier, classifier_features = (None, None)
    if CLASSIFIER_PATH.exists():
        classifier, classifier_features = load_classifier(str(CLASSIFIER_PATH))

    state["master_df"] = master_df
    state["classifier"] = classifier
    state["classifier_features"] = classifier_features
    yield
    state.clear()


app = FastAPI(title="TU CSIT College Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    rank: int = Field(..., gt=0, description="Student's CSIT entrance exam rank (1 = best)")
    preferences: List[str] = Field(
        ..., min_length=1,
        description="Ordered list of preferred college names, as typed by the student"
    )


def _records(df: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe list of dicts (NaN -> null, numpy dtypes -> native)."""
    return json.loads(df.to_json(orient="records"))


def _record(series: pd.Series) -> dict:
    return _records(pd.DataFrame([series]))[0]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/colleges")
def list_colleges():
    """College dropdown data: one entry per college, most competitive first."""
    df = state["master_df"].sort_values("College_Master_Rank")
    cols = [c for c in [
        "College", "College_Master_Rank", "Rank_Category",
        "Latest_Cutoff_Rank", "Predicted_Cutoff_Rank", "Avg_Cutoff_Rank_AllYears",
    ] if c in df.columns]
    return _records(df[cols])


@app.post("/recommend")
def recommend(req: RecommendRequest):
    master_df = state["master_df"]
    classifier = state["classifier"]
    classifier_features = state["classifier_features"]

    ranked_prefs, source_row = recommend_from_preference_list(
        req.rank, req.preferences, master_df, classifier, classifier_features
    )
    top5, top5_note = top5_by_cutoff(req.rank, master_df)
    top5 = attach_admission_probability(top5, req.rank, classifier, classifier_features)

    valid = ranked_prefs.dropna(subset=["Admission_Probability"])
    best_college = _record(valid.iloc[0]) if not valid.empty else None

    return {
        "source_college": _record(source_row),
        "preferences": _records(ranked_prefs),
        "best_college": best_college,
        "top5_recommendations": _records(top5),
        "top5_note": top5_note,
    }


# Mounted last so it doesn't shadow the API routes above — StaticFiles only
# catches paths that don't match an already-registered route.
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
