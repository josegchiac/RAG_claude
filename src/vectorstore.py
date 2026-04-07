"""Gestiona la base de datos vectorial con ChromaDB."""

import chromadb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedder = SentenceTransformer(EMBEDDING_MODEL)
_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))


def _get_collection() -> chromadb.Collection:
    return _chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed(texts: list[str]) -> list[list[float]]:
    vectors = _embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed(texts: list[str]) -> list[list[float]]:
    """Wrapper publico de _embed para uso en otros modulos."""
    return _embed(texts)


# --------------------------------------------------------------------------- #
# Escritura                                                                     #
# --------------------------------------------------------------------------- #

def add_chunks(chunks: list[dict]) -> int:
    """Indexa una lista de chunks en ChromaDB.

    Cada chunk debe tener al menos: text, chunk_id.
    El resto de campos se almacenan como metadata.
    Retorna el numero de chunks insertados.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    # Enriquecer el texto que se embeddea con metadata semantica relevante.
    # El documento almacenado en ChromaDB (texts) no cambia — solo el vector.
    EMBED_FIELDS = ("tema_principal", "responsable", "proceso")
    texts_to_embed = []
    for c in chunks:
        prefix_parts = [
            f"{field.replace('_', ' ').title()}: {c[field]}"
            for field in EMBED_FIELDS
            if c.get(field)
        ]
        prefix = "\n".join(prefix_parts)
        texts_to_embed.append(f"{prefix}\n\n{c['text']}" if prefix else c["text"])

    embeddings = _embed(texts_to_embed)

    # Todo excepto text y chunk_id va como metadata
    # ChromaDB solo acepta str, int, float o bool en metadata
    metadatas = []
    for c in chunks:
        meta = {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
        # Convierte None a string vacio para compatibilidad con ChromaDB
        meta = {k: ("" if v is None else v) for k, v in meta.items()}
        metadatas.append(meta)

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


# --------------------------------------------------------------------------- #
# Busqueda                                                                      #
# --------------------------------------------------------------------------- #

def search(
    query: str,
    n_results: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """Busca los chunks mas relevantes para una consulta.

    Args:
        query:     Pregunta o texto a buscar.
        n_results: Numero de resultados a retornar.
        filters:   Filtros de metadata, ej: {"categoria": "RRHH"}.

    Returns:
        Lista de dicts con text, score y metadata de cada resultado.
    """
    collection = _get_collection()
    query_embedding = _embed([query])[0]

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        if len(filters) == 1:
            kwargs["where"] = filters
        else:
            kwargs["where"] = {"$and": [{k: v} for k, v in filters.items()]}

    results = collection.query(**kwargs)

    output = []
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "chunk_id": chunk_id,
            "text": doc,
            "score": round(1 - dist, 4),   # distancia coseno → similitud
            **meta,
        })

    return output


# --------------------------------------------------------------------------- #
# Utilidades                                                                    #
# --------------------------------------------------------------------------- #

def collection_info() -> dict:
    """Retorna informacion basica de la coleccion."""
    collection = _get_collection()
    return {
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "chroma_path": str(CHROMA_PATH),
    }


def get_by_ids(chunk_ids: list[str]) -> list[dict]:
    """Recupera chunks especificos por sus IDs de ChromaDB.

    Usado por el retrieval basado en grafo para obtener los chunks
    vinculados a las entidades encontradas.
    """
    if not chunk_ids:
        return []
    collection = _get_collection()
    results = collection.get(
        ids=chunk_ids,
        include=["documents", "metadatas"],
    )
    output = []
    for chunk_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        output.append({"chunk_id": chunk_id, "text": doc, "score": None, **meta})
    return output


def delete_by_source(source: str) -> None:
    """Elimina todos los chunks de un documento especifico."""
    collection = _get_collection()
    collection.delete(where={"source": source})


def get_metadata_values(fields: list[str] | None = None) -> dict[str, list[str]]:
    """Retorna los valores unicos disponibles para campos filtrables.

    Args:
        fields: Campos a inspeccionar. Por defecto: categoria, proceso, source.

    Returns:
        Dict campo -> lista de valores unicos no vacios, ordenados alfabeticamente.
    """
    if fields is None:
        fields = ["proceso", "source"]

    collection = _get_collection()
    results = collection.get(include=["metadatas"])

    values: dict[str, set] = {f: set() for f in fields}
    for meta in results["metadatas"]:
        for field in fields:
            val = meta.get(field, "")
            if val:
                values[field].add(val)

    return {k: sorted(v) for k, v in values.items()}
