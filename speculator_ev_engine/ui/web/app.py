"""FastAPI application serving speculator_ev_engine computations."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from speculator_ev_engine.ui.web.routes.kelly import router as kelly_router
from speculator_ev_engine.ui.web.routes.icm import router as icm_router
from speculator_ev_engine.ui.web.routes.sports import router as sports_router
from speculator_ev_engine.ui.web.routes.decisions import router as decisions_router

app = FastAPI(
    title="speculator_ev_engine",
    version="0.1.0",
    description="EV-maximizing framework — API layer",
)

app.include_router(kelly_router)
app.include_router(icm_router)
app.include_router(sports_router)
app.include_router(decisions_router)

# Serve static frontend
_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8269)


if __name__ == "__main__":
    main()
