"""Deploy inference API ke Hugging Face Spaces (Docker, gratis).

Pemakaian (dari root repo, setelah commit):
    python scripts/deploy_hf_space.py [--space nama-space]

Butuh di .env: HF_TOKEN (token Hugging Face dengan akses write). GEMINI_API_KEY dan
DATABASE_URL dari .env dikirim sebagai Secrets Space (tidak ikut diunggah sebagai file).

Yang diunggah hanya file yang dilacak git (`git ls-files`, versi HEAD), sehingga .env,
.venv, dan data tersembunyi tidak mungkin ikut. README Space diberi header konfigurasi
(sdk: docker, app_port: 7860) di atas README repo.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ("GEMINI_API_KEY", "DATABASE_URL")
SKIP_PREFIXES = ("notebooks/",)  # tidak dibutuhkan API

SPACE_HEADER = """---
title: Firaza Vendor Recommender API
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Inference API rekomendasi vendor (FastAPI)
---

> Inference API Firaza Vendor Recommender. Dokumentasi interaktif: buka `/docs`.
> Kode sumber: https://github.com/broBaagas/firaza-vendor-recommender

"""


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and not f.startswith(SKIP_PREFIXES)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="firaza-api", help="nama Space (default firaza-api)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN belum diisi di .env")
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("Peringatan: ada perubahan yang belum di-commit; yang diunggah adalah isi file saat ini "
              "untuk file yang dilacak git.")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    print(f"Space: https://huggingface.co/spaces/{repo_id}")

    files = tracked_files()
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)) for f in files if f != "README.md"]
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))

    keep = set(files) | {"README.md", ".gitattributes"}
    stale = [f for f in api.list_repo_files(repo_id, repo_type="space") if f not in keep]
    ops += [CommitOperationDelete(path_in_repo=f) for f in stale]

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    api.create_commit(repo_id, operations=ops, repo_type="space", commit_message=f"Deploy dari GitHub {head}")
    print(f"Diunggah {len(ops) - len(stale)} file, dihapus {len(stale)} file lama.")

    for key in SECRETS:
        value = os.environ.get(key, "").strip()
        if value:
            api.add_space_secret(repo_id, key, value)
            print(f"Secret {key} diatur.")
        else:
            print(f"Secret {key} kosong di .env, dilewati.")

    subdomain = repo_id.replace("/", "-").replace("_", "-").lower()
    print(f"API (setelah build selesai, ±5 menit): https://{subdomain}.hf.space/docs")


if __name__ == "__main__":
    main()
