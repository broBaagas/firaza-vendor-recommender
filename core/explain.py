"""Penjelasan rekomendasi dengan Gemini (google-genai).

Gemini hanya MENJELASKAN, tidak menentukan ranking. Setiap angka di penjelasan
divalidasi terhadap konteks; jika gagal atau kuota habis, dipakai template cadangan.
"""

import os

GEMINI_MODEL = "gemini-2.5-flash"


def build_context(need: dict, top: list[dict]) -> dict:
    """Konteks JSON: kebutuhan + 3 teratas + angka-angkanya."""
    raise NotImplementedError


def explain_with_gemini(context: dict) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    raise NotImplementedError


def validate_numbers(text: str, context: dict) -> bool:
    """True jika semua angka di teks ada di konteks."""
    raise NotImplementedError


def fallback_explanation(context: dict) -> str:
    raise NotImplementedError


def explain(need: dict, top: list[dict]) -> dict:
    """Kembalikan {"teks": ..., "sumber": "gemini" | "template"}."""
    raise NotImplementedError
