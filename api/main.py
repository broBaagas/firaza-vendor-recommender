"""Inference API. Jalankan: uvicorn api.main:app --reload"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import MODEL_VERSION
from core.ranking import PRESETS

app = FastAPI(title="Firaza Vendor Recommender API", version=MODEL_VERSION)


class RecommendRequest(BaseModel):
    material_id: str
    qty: int = Field(gt=0)
    hari_tersedia: int = Field(gt=0, description="Jumlah hari sampai deadline")
    budget: float | None = Field(default=None, gt=0)
    preset: str = "Seimbang"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/api/v1/recommend")
def recommend(req: RecommendRequest) -> dict:
    if req.preset not in PRESETS:
        raise HTTPException(status_code=422, detail=f"preset harus salah satu dari {list(PRESETS)}")
    raise HTTPException(status_code=501, detail="Belum diimplementasikan (Minggu 3)")
