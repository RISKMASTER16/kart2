"""Local web server for the interactive demo and the backtest results.

    pip install -r requirements.txt
    uvicorn api:app --port 8000
    # then open http://127.0.0.1:8000/

Pages and files:
    GET /                      redirects to the front end
    GET /demo/                 the front end (demo/)
    GET /demo.json             the replay file read by the front end
    GET /out/<file>            backtest outputs (results.csv, hourly.csv, charts, reports)

JSON API (files are read on each request, so re-running backtest.py needs no restart):
    GET /api/demo              the whole replay file (INTERFACES.md section 5)
    GET /api/hours             replay hours with both hit rates
    GET /api/hours/{i}         one replay hour: top pages and whether each method had them cached
    GET /api/spikes            spikes on the replay day
    GET /api/summary           summary hit rates and the foresight curve
    GET /api/results           out/results.csv as JSON (optional ?horizon=1&cache_size=100000)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DEMO_DIR = ROOT / "demo"

app = FastAPI(title="Ahead of the Spike")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def replay_path() -> Path:
    """The published replay file at the repository root, or a fresh one from backtest.py."""
    for path in (ROOT / "demo.json", OUT / "demo.json"):
        if path.exists():
            return path
    raise HTTPException(503, "demo.json not found; run python backtest.py first")


def demo() -> dict:
    return json.loads(replay_path().read_text(encoding="utf-8"))


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/demo/#start" if DEMO_DIR.is_dir() else "/docs")


@app.get("/demo.json", include_in_schema=False)
def demo_file():
    return FileResponse(replay_path(), media_type="application/json")


@app.get("/api/demo")
def get_demo():
    return demo()


@app.get("/api/hours")
def get_hours():
    d = demo()
    return {"day": d["day"], "predictors": d["predictors"], "our_model_name": d.get("our_model_name"),
            "cache_size": d["cache_size"],
            "hours": [{"i": i, "hour": h["hour"], "hit_rate": h["hit_rate"], "total_views": h["total_views"]}
                      for i, h in enumerate(d["hours"])]}


@app.get("/api/hours/{i}")
def get_hour(i: int):
    hours = demo()["hours"]
    if not 0 <= i < len(hours):
        raise HTTPException(404, f"hour index must be 0..{len(hours) - 1}")
    return hours[i]


@app.get("/api/spikes")
def get_spikes():
    return demo()["spikes"]


@app.get("/api/summary")
def get_summary():
    d = demo()
    return {"our_model_name": d.get("our_model_name"), **d["summary"]}


@app.get("/api/results")
def get_results(horizon: int | None = None, cache_size: int | None = None):
    path = OUT / "results.csv"
    if not path.exists():
        raise HTTPException(503, "out/results.csv not found; run python backtest.py first")
    df = pd.read_csv(path)
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    if cache_size is not None:
        df = df[df["cache_size"] == cache_size]
    return json.loads(df.to_json(orient="records"))


if OUT.is_dir():
    app.mount("/out", StaticFiles(directory=OUT), name="out")
    if (OUT / "charts").is_dir():
        app.mount("/charts", StaticFiles(directory=OUT / "charts"), name="charts")
if DEMO_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=DEMO_DIR, html=True), name="demo")
