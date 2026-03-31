"""Carga y divide documentos PDF y DOCX en chunks con metadatos hibridos."""

import re
import base64
import json
import tiktoken
import anthropic
import pdfplumber
from datetime import date
from pathlib import Path
from docx import Document


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUMMARY_TOKENS = 100   # tokens del inicio del doc usados como resumen de contexto
CONFIG_PATH = Path(__file__).parent.parent / "docs_config.json"

_enc = tiktoken.get_encoding("cl100k_base")
_client = anthropic.Anthropic()


# --------------------------------------------------------------------------- #
# Limpieza de texto                                                             #
# --------------------------------------------------------------------------- #

def _detect_boilerplate(pages_text: list[str], threshold: float = 0.7) -> list[str]:
    """Detecta lineas que se repiten en mas del threshold de paginas.

    Retorna la lista de lineas consideradas boilerplate (headers/footers).
    """
    if not pages_text:
        return []

    # Cuenta en cuantas paginas aparece cada linea no vacia
    line_counts: dict[str, int] = {}
    for text in pages_text:
        seen = set()
        for line in text.splitlines():
            line = line.strip()
            if line and line not in seen:
                line_counts[line] = line_counts.get(line, 0) + 1
                seen.add(line)

    min_pages = max(2, int(len(pages_text) * threshold))
    return [line for line, count in line_counts.items() if count >= min_pages]


def _clean_text(text: str, boilerplate: list[str]) -> str:
    """Elimina lineas boilerplate y normaliza espacios del texto extraido."""
    lines = []
    for line in text.splitlines():
        if line.strip() not in boilerplate:
            lines.append(line)
    text = "\n".join(lines)

    # "palabra\n palabra" producido por PDFs con saltos de linea mid-frase
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Espacios multiples → uno solo
    text = re.sub(r' {2,}', ' ', text)
    # Lineas de numero de pagina residuales
    text = re.sub(r'\bPage\s+\d+\s*/\s*\d+\b', '', text, flags=re.IGNORECASE)

    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking                                                                      #
# --------------------------------------------------------------------------- #

def _split_text(text: str) -> list[str]:
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunks.append(_enc.decode(tokens[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _extract_summary(full_text: str) -> str:
    """Toma los primeros SUMMARY_TOKENS tokens del documento como resumen de contexto."""
    tokens = _enc.encode(full_text)
    return _enc.decode(tokens[:SUMMARY_TOKENS])


# --------------------------------------------------------------------------- #
# Metadatos                                                                     #
# --------------------------------------------------------------------------- #

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def _semantic_metadata_from_claude(text: str, filename: str, doc_summary: str, base_config: dict | None = None) -> dict:
    """Llama a Claude Haiku para extraer metadatos semanticos de un chunk.

    Recibe el nombre del archivo, el inicio del documento y la metadata base
    del config como contexto para generar metadata especifica del chunk.
    """
    config_context = ""
    if base_config:
        config_context = (
            f"Contexto del documento completo:\n"
            f"- Tema general: {base_config.get('tema_principal', '')}\n"
            f"- Categoría: {base_config.get('categoria', '')}\n"
            f"- Proceso: {base_config.get('proceso', '')}\n"
            f"- Responsable general: {base_config.get('responsable', '')}\n\n"
        )
    prompt = (
        f"Nombre del documento: {filename}\n\n"
        f"{config_context}"
        f"Inicio del documento:\n{doc_summary}\n\n"
        "---\n"
        f"Fragmento a analizar:\n{text}\n\n"
        "Basándote en el fragmento específico (no en el documento completo), "
        "responde UNICAMENTE con un objeto JSON valido:\n"
        '{"tema_principal": "...", "categoria": "...", "proceso": "...", "responsable": "... o null"}'
    )
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    result = json.loads(raw[start:end])
    if "categoria" in result and result["categoria"]:
        result["categoria"] = result["categoria"].strip().title()
    return result


def _build_metadata(filename: str, page: int | None, chunk_index: int) -> dict:
    """Construye metadatos estructurales del chunk."""
    return {
        "source": filename,
        "page": page,
        "chunk_id": f"{Path(filename).stem}_p{page}_c{chunk_index}",
        "file_type": Path(filename).suffix.lstrip("."),
        "indexed_at": date.today().isoformat(),
    }


# --------------------------------------------------------------------------- #
# Extraccion de PDF                                                             #
# --------------------------------------------------------------------------- #

_VISION_PROMPT = (
    "Extrae TODO el contenido de esta página en texto plano.\n"
    "Para tablas: una fila por línea, columnas separadas por |\n"
    "Para matrices RACI: 'Actividad X — R: rol, A: rol, C: rol, I: rol'\n"
    "Para diagramas/flujogramas: describe el flujo en pasos numerados\n"
    "Para imágenes: una línea describiendo qué muestra\n"
    "Solo el contenido extraído, sin comentarios ni explicaciones."
)


def _needs_vision(page: "pdfplumber.page.Page") -> bool:
    """Decide si una pagina requiere Claude Vision para extraccion correcta."""
    # Condicion 1: tiene imagenes embebidas reales (excluye lineas decorativas)
    real_images = [
        img for img in page.images
        if img.get("width", 0) > 10 and img.get("height", 0) > 10
    ]
    if real_images:
        return True

    # Condicion 4: texto muy corto vs area de pagina (contenido visual)
    text = page.extract_text() or ""
    area = (page.width or 1) * (page.height or 1)
    if len(text) < (area / 100_000) * 50:
        return True

    tables = page.extract_tables() or []
    for table in tables:
        for row in table:
            if not row:
                continue
            # Condicion 2: celdas combinadas (None) > 60% en alguna fila
            none_ratio = sum(1 for c in row if c is None) / len(row)
            if none_ratio > 0.6:
                return True
            # Condicion 3: fila con una sola celda de mas de 100 chars
            non_none = [c for c in row if c is not None]
            if len(non_none) == 1 and len(non_none[0]) > 100:
                return True

    return False


def _extract_page_with_vision(page: "pdfplumber.page.Page") -> str:
    """Renderiza la pagina como PNG y extrae contenido con Claude Haiku Vision."""
    img = page.to_image(resolution=150)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
    )
    return response.content[0].text.strip()


def _extract_page_with_pdfplumber(page: "pdfplumber.page.Page") -> str:
    """Extrae texto y tablas de una pagina con pdfplumber."""
    parts = []

    text = page.extract_text() or ""
    if text.strip():
        parts.append(text.strip())

    tables = page.extract_tables() or []
    for table in tables:
        rows = []
        for row in table:
            row_text = " | ".join(cell.strip() if cell else "" for cell in row)
            if row_text.replace("|", "").strip():
                rows.append(row_text)
        if rows:
            parts.append("\n".join(rows))

    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Loaders                                                                       #
# --------------------------------------------------------------------------- #

def load_pdf(path: str, enrich: bool = True, use_vision: bool = True) -> list[dict]:
    config = _load_config()
    filename = Path(path).name

    vision_pages = 0
    plumber_pages = 0

    with pdfplumber.open(path) as pdf:
        # Primera pasada: extraer texto para doc_summary y deteccion de boilerplate
        pages_raw: list[str] = []
        for page in pdf.pages:
            try:
                pages_raw.append(_extract_page_with_pdfplumber(page))
            except Exception:
                pages_raw.append(page.extract_text() or "")

        doc_summary = _extract_summary("\n".join(pages_raw))
        boilerplate = _detect_boilerplate(pages_raw)
        base_config = config.get(filename)

        result = []
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                if use_vision and _needs_vision(page):
                    raw_text = _extract_page_with_vision(page)
                    vision_pages += 1
                else:
                    raw_text = pages_raw[page_num - 1]
                    plumber_pages += 1
            except Exception as e:
                print(f"  [Vision error p{page_num}: {e} — fallback a pdfplumber]")
                raw_text = pages_raw[page_num - 1]
                plumber_pages += 1

            text = _clean_text(raw_text, boilerplate)
            for i, chunk in enumerate(_split_text(text)):
                if not chunk.strip():
                    continue
                metadata = _build_metadata(filename, page_num, i)
                if enrich:
                    semantic = _semantic_metadata_from_claude(chunk, filename, doc_summary, base_config)
                    metadata.update({"semantic_source": "claude", **semantic})
                result.append({"text": chunk, **metadata})

    print(f"Vision: {vision_pages} páginas | pdfplumber: {plumber_pages} páginas")
    return result


def load_docx(path: str, enrich: bool = True) -> list[dict]:
    config = _load_config()
    filename = Path(path).name
    doc = Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    doc_summary = _extract_summary(full_text)

    base_config = config.get(filename)
    result = []
    for i, chunk in enumerate(_split_text(full_text)):
        if not chunk.strip():
            continue
        metadata = _build_metadata(filename, None, i)
        if enrich:
            semantic = _semantic_metadata_from_claude(chunk, filename, doc_summary, base_config)
            metadata.update({"semantic_source": "claude", **semantic})
        result.append({"text": chunk, **metadata})

    return result


def load_document(path: str, enrich: bool = True) -> list[dict]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, enrich)
    elif ext == ".docx":
        return load_docx(path, enrich)
    else:
        raise ValueError(f"Formato no soportado: {ext}")
