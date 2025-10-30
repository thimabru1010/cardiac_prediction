import os
import io
import csv
import sys
import time
import math
import json
import random
import argparse
import threading
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from tqdm import tqdm

# =========================
# Configurações
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
# Export padrão para Google Workspace (ajuste se quiser)
EXPORT_MIME_MAP: Dict[str, str] = {
    "application/vnd.google-apps.document": "application/pdf",
    "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.google-apps.drawing": "image/png",
}
EXPORT_EXT_MAP: Dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "image/png": ".png",
}

CHUNK_SIZE = 1024 * 1024 * 8  # 8MB por chunk
MAX_WORKERS = max(8, os.cpu_count() or 4)  # paralelismo padrão
RETRY_ATTEMPTS = 5
LOG_CSV = "download_log.csv"

print_lock = threading.Lock()
def log_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# =========================
# Autenticação / Service
# =========================
def get_drive_service() -> any: #type: ignore
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                log_print("ERRO: coloque credentials.json (OAuth) na pasta do script.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            
            # Modificação para servidor remoto (SSH)
            # Em vez de run_local_server, usa run_console para autenticação manual
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            log_print("\n" + "="*70)
            log_print("AUTENTICAÇÃO NECESSÁRIA")
            log_print("="*70)
            log_print("\n1. Abra este link no seu navegador LOCAL:\n")
            log_print(f"   {auth_url}\n")
            log_print("2. Faça login e autorize o aplicativo")
            log_print("3. Copie o código de autorização que aparecerá")
            log_print("4. Cole o código abaixo:\n")
            log_print("="*70 + "\n")
            
            code = input("Cole o código de autorização aqui: ").strip()
            
            flow.fetch_token(code=code)
            creds = flow.credentials
            
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds), creds

def ensure_token(creds: Credentials) -> str:
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds.token

# =========================
# Utilidades de Drive
# =========================
def extract_drive_id(input_str: str) -> str:
    """
    Extrai o ID do Google Drive a partir de um link compartilhável ou retorna o próprio ID.
    
    Formatos suportados:
    - https://drive.google.com/drive/folders/ID
    - https://drive.google.com/drive/folders/ID?usp=sharing
    - https://drive.google.com/file/d/ID/view
    - https://drive.google.com/file/d/ID/view?usp=sharing
    - https://drive.google.com/open?id=ID
    - ID direto
    """
    input_str = input_str.strip()
    
    # Padrões de regex para diferentes formatos de link
    patterns = [
        r'/folders/([a-zA-Z0-9_-]+)',  # pasta
        r'/file/d/([a-zA-Z0-9_-]+)',    # arquivo
        r'[?&]id=([a-zA-Z0-9_-]+)',     # open?id=
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return match.group(1)
    
    # Se não encontrou padrão, assume que já é um ID
    # Valida se tem formato básico de ID do Drive (letras, números, _ e -)
    if re.match(r'^[a-zA-Z0-9_-]+$', input_str):
        return input_str
    
    raise ValueError(f"Não foi possível extrair ID do Drive de: {input_str}")

def resolve_shortcut(service, fobj: dict) -> dict:
    if fobj.get("mimeType") == SHORTCUT_MIME and "shortcutDetails" in fobj:
        tid = fobj["shortcutDetails"]["targetId"]
        return service.files().get(
            fileId=tid,
            fields="id, name, mimeType, size",
            supportsAllDrives=True,
        ).execute()
    return fobj

def collect_tasks(service, folder_id: str, root_out: Path) -> List[Tuple[str, str, str, Optional[int], Path]]:
    tasks = []
    def _walk(fid: str, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        q = f"'{fid}' in parents and trashed=false"
        page_token = None
        while True:
            resp = service.files().list(
                q=q,
                pageSize=1000,
                pageToken=page_token,
                fields="nextPageToken, files(id,name,mimeType,size,shortcutDetails)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            for f in resp.get("files", []):
                f2 = resolve_shortcut(service, f)
                fid2 = f2["id"]
                name2 = f2["name"]
                mime2 = f2["mimeType"]
                size2 = int(f2["size"]) if "size" in f2 else None
                if mime2 == FOLDER_MIME:
                    _walk(fid2, out_dir / name2)
                else:
                    tasks.append((fid2, name2, mime2, size2, out_dir))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    _walk(folder_id, root_out)
    return tasks

# =========================
# Download com Range (resume)
# =========================
# def download_binary_with_range(file_id: str, name: str, size: Optional[int], out_dir: Path, token: str):
#     """
#     Faz GET em https://www.googleapis.com/drive/v3/files/{id}?alt=media
#     Usa Authorization: Bearer e Range: bytes=offset-
#     Retoma se arquivo parcial existir.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     dest = out_dir / name
#     url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
#     headers = {"Authorization": f"Bearer {token}"}

#     # Checa progresso local
#     existing = dest.stat().st_size if dest.exists() else 0
#     if size and existing >= size:
#         log_print(f"✅ Já concluído: {dest}")
#         return dest

#     # Se já existe parcial, abre em append e usa Range
#     mode = "ab" if existing > 0 else "wb"
#     if existing > 0:
#         headers["Range"] = f"bytes={existing}-"
#         log_print(f"↻ Retomando: {name} ({existing}/{size or '??'} bytes)")

#     with requests.get(url, headers=headers, stream=True, timeout=120) as r:
#         if r.status_code in (200, 206):  # 206 = Partial Content (Range aceito)
#             with open(dest, mode) as f:
#                 for chunk in r.iter_content(CHUNK_SIZE):
#                     if chunk:
#                         f.write(chunk)
#         else:
#             # Propaga erro para camada de retry
#             raise RuntimeError(f"HTTP {r.status_code} ao baixar {name}")

#     # Validação simples de tamanho
#     if size is not None and dest.stat().st_size != size:
#         log_print(f"⚠️ Tamanho divergente em {dest.name}: {dest.stat().st_size} vs {size}")
#     return dest

def download_binary_with_range(file_id: str, name: str, size: Optional[int], out_dir: Path, token: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {token}"}

    existing = dest.stat().st_size if dest.exists() else 0
    if size and existing >= size:
        log_print(f"✅ Já concluído: {dest}")
        return dest

    mode = "ab" if existing > 0 else "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        log_print(f"↻ Retomando: {name} ({existing}/{size or '??'} bytes)")

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} ao baixar {name}")

        total = None
        if size is not None:
            # quando há Range (206), o servidor pode não mandar Content-Length total; então a barra soma existing
            total = size

        # barra de progresso por arquivo
        with open(dest, mode) as f, tqdm(
            total=total,
            initial=existing if (total is not None) else 0,
            unit="B",
            unit_scale=True,
            desc=name,
            leave=False,      # evita poluir saída com muitas barras
        ) as pbar:
            for chunk in r.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # checagem simples
    if size is not None and dest.stat().st_size != size:
        log_print(f"⚠️ Tamanho divergente em {dest.name}: {dest.stat().st_size} vs {size}")
    else:
        log_print(f"✅ {name}")
    return dest

def export_google_file(file_id: str, name: str, src_mime: str, out_dir: Path, token: str):
    export_mime = EXPORT_MIME_MAP[src_mime]
    ext = EXPORT_EXT_MAP[export_mime]
    dest = out_dir / f"{name}{ext}"
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"mimeType": export_mime}

    if dest.exists():
        log_print(f"✅ (export) Já existe: {dest.name}")
        return dest

    with requests.get(url, headers=headers, params=params, stream=True, timeout=120) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} ao exportar {name}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest

# =========================
# Worker com retries + logging
# =========================
def backoff_sleep(attempt: int):
    time.sleep((2 ** attempt) + random.random())

def process_task(service, creds: Credentials, row: Tuple[str, str, str, Optional[int], Path], log_q: list):
    file_id, name, mime, size, out_dir = row
    start = time.time()
    status = "ok"
    local_path = ""
    try:
        token = ensure_token(creds)
        if mime in EXPORT_MIME_MAP:
            local_path = str(export_google_file(file_id, name, mime, out_dir, token))
        else:
            # retries p/ 403/429/5xx
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    local_path = str(download_binary_with_range(file_id, name, size, out_dir, token))
                    break
                except RuntimeError as e:
                    # quando é erro de quota/limite, tente esperar e repetir
                    if any(code in str(e) for code in ["HTTP 403", "HTTP 429", "HTTP 500", "HTTP 503"]):
                        log_print(f"Erro temporário em {name}: {e}. Tentativa {attempt+1}/{RETRY_ATTEMPTS}")
                        backoff_sleep(attempt)
                        continue
                    raise
            else:
                status = "failed-retries"
    except HttpError as e:
        status = f"http-error-{getattr(e.resp, 'status', 'unknown')}"
    except Exception as e:
        status = f"error-{type(e).__name__}: {e}"
    finally:
        end = time.time()
        log_q.append({
            "file_id": file_id,
            "name": name,
            "mimeType": mime,
            "size": size or "",
            "local_path": local_path,
            "status": status,
            "start_ts": int(start),
            "end_ts": int(end),
            "elapsed_s": round(end - start, 3),
        })
        if status == "ok":
            log_print(f"✅ {name}")
        else:
            log_print(f"🚨 {name} -> {status}")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Baixar pasta compartilhada do Google Drive (paralelo + resume + log)")
    ap.add_argument("folder_input", help="ID ou link compartilhável da pasta/arquivo no Google Drive")
    ap.add_argument("-o", "--output", default=".", help="Diretório base para salvar")
    ap.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="Número de downloads em paralelo")
    args = ap.parse_args()

    # Extrai o ID do link ou usa o ID direto
    try:
        folder_id = extract_drive_id(args.folder_input)
        log_print(f"ID extraído: {folder_id}")
    except ValueError as e:
        log_print(f"ERRO: {e}")
        sys.exit(1)

    service, creds = get_drive_service()
    meta = service.files().get(
        fileId=folder_id,
        fields="id,name,mimeType",
        supportsAllDrives=True
    ).execute()

    if meta["mimeType"] != FOLDER_MIME:
        log_print("ERRO: O ID informado não é de uma pasta.")
        sys.exit(1)

    root_out = Path(args.output).expanduser().resolve() / meta["name"]
    log_print(f"Baixando pasta '{meta['name']}' para '{root_out}' …")

    tasks = collect_tasks(service, folder_id, root_out)
    log_print(f"Total de arquivos: {len(tasks)}")

    log_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(process_task, service, creds, t, log_rows) for t in tasks]
        for f in as_completed(futs):
            _ = f.result()  # força levantar exceções aqui, se houver

    # salvar log CSV
    if log_rows:
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        log_print(f"📄 Log salvo em: {LOG_CSV}")

    log_print("✅ Concluído.")

if __name__ == "__main__":
    main()
