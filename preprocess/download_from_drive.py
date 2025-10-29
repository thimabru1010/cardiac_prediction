import argparse
import re
import gdown
from pathlib import Path

# Extrai o ID da URL (funciona com /file/d/<id>/view, open?id=<id>, etc.)
def extract_id(s: str) -> str:
    for pat in [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)"
    ]:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return s.strip()  # já é o ID

def main():
    p = argparse.ArgumentParser(description="Baixa 1 arquivo público do Google Drive")
    p.add_argument("url_or_id", help="URL completa do Drive ou apenas o file ID")
    p.add_argument("-o", "--output", default=".", help="caminho do arquivo de saída (ou diretório)")
    p.add_argument("--resume", action="store_true", help="retomar download interrompido")
    args = p.parse_args()

    file_id = extract_id(args.url_or_id)
    url = f"https://drive.google.com/uc?id={file_id}"

    # Se 'output' for um diretório, gdown usará o nome original do arquivo
    out = str(Path(args.output))

    print(f"Baixando: {url}")
    gdown.download(url=url, output=out, quiet=False, fuzzy=True, resume=args.resume)

if __name__ == "__main__":
    main()
