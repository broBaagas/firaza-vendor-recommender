"""Inference API Firaza Vendor Recommender.

Jalankan lokal: uvicorn api.main:app --reload  (dokumentasi interaktif di /docs)

Endpoint utama POST /api/v1/recommend memakai logika yang sama dengan Streamlit
(core.ranking.recommend untuk ranking, core.explain.explain untuk penjelasan).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, model_validator

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # lokal; di Render pakai env var

from core import MODEL_VERSION, storage  # noqa: E402
from core.data import load_data  # noqa: E402
from core.explain import explain  # noqa: E402
from core.ranking import load_model, recommend as core_recommend  # noqa: E402

app = FastAPI(
    title="Firaza Vendor Recommender API",
    version=MODEL_VERSION,
    description=(
        "Rekomendasi vendor untuk satu kebutuhan pengadaan. Model ML memprediksi peluang berhasil "
        "(datang tepat deadline dan cacat ≤ 3%) per vendor, lalu vendor diurutkan dengan skor berbobot "
        "(peluang berhasil, harga, kecepatan). Gemini hanya menulis penjelasan; ranking tidak berubah. "
        "Data vendor dan transaksi sintetis."
    ),
)


@lru_cache(maxsize=1)
def get_data() -> dict:
    return load_data()


@lru_cache(maxsize=1)
def get_model():
    return load_model()


# ---------------------------------------------------------------------------
# Skema
# ---------------------------------------------------------------------------

class Bobot(BaseModel):
    w_berhasil: float = Field(ge=0, description="Bobot peluang berhasil")
    w_harga: float = Field(ge=0, description="Bobot harga murah")
    w_cepat: float = Field(ge=0, description="Bobot lead time cepat")


class RecommendRequest(BaseModel):
    material_id: str = Field(description="Kode material, lihat GET /api/v1/materials", examples=["M01"])
    qty: int = Field(gt=0, description="Jumlah yang dibutuhkan", examples=[500])
    hari_tersedia: int = Field(gt=0, description="Jumlah hari sampai deadline", examples=[14])
    budget: float | None = Field(default=None, gt=0, description="Batas total harga (Rp), opsional")
    preset: Literal["Seimbang", "Termurah", "Tercepat", "Kustom"] = "Seimbang"
    bobot: Bobot | None = Field(default=None, description="Wajib jika preset = Kustom; dinormalisasi agar berjumlah 1")
    top_k: int = Field(default=3, ge=1, le=10, description="Jumlah vendor teratas di field 'top'")
    dengan_penjelasan: bool = Field(default=True, description="Sertakan penjelasan (Gemini atau template)")

    @model_validator(mode="after")
    def _check_bobot(self):
        if self.preset == "Kustom":
            if self.bobot is None:
                raise ValueError("preset Kustom membutuhkan field 'bobot'")
            if self.bobot.w_berhasil + self.bobot.w_harga + self.bobot.w_cepat <= 0:
                raise ValueError("bobot tidak boleh semuanya 0")
        elif self.bobot is not None:
            raise ValueError("field 'bobot' hanya dipakai jika preset = Kustom")
        return self


class VendorHasil(BaseModel):
    peringkat: int
    vendor_id: str
    nama_vendor: str
    kota: str
    offer_id: str
    harga_satuan: float
    total_harga: float
    lead_time_janji: int
    moq: int
    p_berhasil: float = Field(description="Prediksi peluang berhasil (0-1)")
    skor: float
    jumlah_transaksi: int
    data_terbatas: bool = Field(description="True jika rekam jejak < 5 transaksi")
    tingkat_berhasil_10: float
    rata_hari_telat: float
    tingkat_cacat: float
    berhasil_musim_ramai: float


class Validasi(BaseModel):
    valid: bool
    jumlah_angka: int
    cocok: int
    tidak_cocok: list[str]
    menyebut_peringkat_1: bool


class PenjelasanOut(BaseModel):
    rekomendasi: str
    alasan: list[str]
    trade_off: list[str]
    saran: list[str]
    sumber: Literal["gemini", "template"]
    model: str | None = Field(description="Model Gemini, atau null jika template")
    catatan: str | None = Field(description="Alasan memakai template, jika ada")
    validasi: Validasi


class RecommendResponse(BaseModel):
    kebutuhan: dict
    preset: str
    bobot: dict[str, float]
    model: str = Field(description="Versi model ML yang dipakai")
    jumlah_kandidat: int
    top: list[VendorHasil]
    ranking: list[VendorHasil] = Field(description="Semua kandidat yang lolos saringan, terurut")
    pesan: str | None = Field(default=None, description="Diisi jika tidak ada kandidat")
    penjelasan: PenjelasanOut | None
    riwayat_id: int | None = Field(default=None, description="ID baris di tabel riwayat, atau null jika tidak tersimpan")


class Material(BaseModel):
    material_id: str
    nama_material: str
    kategori: str
    satuan: str
    lead_time_min: int
    lead_time_max: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    model = get_model()
    return {"status": "ok", "app_version": MODEL_VERSION,
            "model_version": model.get("version", "model") if model else "baseline"}


@app.get("/api/v1/materials", response_model=list[Material])
def materials() -> list[dict]:
    cols = list(Material.model_fields)
    return get_data()["materials"][cols].to_dict(orient="records")


@app.post("/api/v1/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> dict:
    """Ranking vendor untuk satu kebutuhan, plus penjelasan.

    Fungsi sinkron (bukan async) agar panggilan Gemini berjalan di threadpool dan tidak memblokir server.
    """
    data = get_data()
    if req.material_id not in set(data["materials"]["material_id"]):
        raise HTTPException(status_code=404, detail=f"material_id '{req.material_id}' tidak dikenal; "
                                                    "lihat GET /api/v1/materials")
    need = {"material_id": req.material_id, "qty": req.qty, "hari_tersedia": req.hari_tersedia,
            "budget": req.budget}
    kustom = req.preset == "Kustom"
    try:
        result = core_recommend(need, preset="Seimbang" if kustom else req.preset, top_k=req.top_k,
                                weights=req.bobot.model_dump() if kustom else None,
                                data=data, model=get_model())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    result["penjelasan"] = explain(result, data["materials"]) if req.dengan_penjelasan else None
    result["riwayat_id"] = storage.safe_save(result, result["penjelasan"], "api")
    return result
