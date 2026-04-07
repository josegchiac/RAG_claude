"""Conecta la busqueda vectorial y el grafo de conocimiento con el LLM."""

import json
import anthropic
from vectorstore import search, get_by_ids, get_metadata_values
import graphstore
from utils import parse_json_response


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
N_CHUNKS = 5  # chunks a recuperar por consulta

_client = anthropic.Anthropic()


SYSTEM_PROMPT = """Eres un asistente corporativo experto en responder preguntas
basandote exclusivamente en la documentacion interna de la empresa.

Reglas:
- Responde SOLO con informacion presente en los fragmentos proporcionados.
- Si la informacion no esta en los fragmentos, indica claramente que no encontraste
  informacion suficiente en la documentacion disponible.
- Cita siempre la fuente (nombre del documento y pagina si esta disponible).
- Sé conciso y directo.
- Si hay contradicciones entre fragmentos, mencionalas."""


# --------------------------------------------------------------------------- #
# Clasificacion de preguntas                                                    #
# --------------------------------------------------------------------------- #



# --------------------------------------------------------------------------- #
# Retrieval                                                                     #
# --------------------------------------------------------------------------- #

def _graph_retrieval(question: str, n_chunks: int) -> list[dict]:
    """Busca chunks a traves del grafo de conocimiento.

    Busca semanticamente el query completo contra los labels del grafo,
    expande el subgrafo de los nodos mas similares y recupera los chunks
    vinculados desde ChromaDB.
    """
    matches = graphstore.search_entities_semantic(question, top_k=5, threshold=0.3)
    if not matches:
        return []

    all_chunk_ids: set[str] = set()
    for match in matches:
        chunk_ids = graphstore.get_subgraph_chunk_ids(match["id"], depth=2)
        all_chunk_ids.update(chunk_ids)

    if not all_chunk_ids:
        return []

    chunks = get_by_ids(list(all_chunk_ids))

    # Marcar origen y limitar cantidad
    for chunk in chunks:
        chunk["retrieval_source"] = "graph"
    return chunks[:n_chunks]


def _vector_retrieval(question: str, n_chunks: int, filters: dict | None) -> list[dict]:
    """Busca chunks por similitud vectorial en ChromaDB."""
    chunks = search(question, n_results=n_chunks, filters=filters)
    for chunk in chunks:
        chunk["retrieval_source"] = "vector"
    return chunks


def _merge_chunks(graph_chunks: list[dict], vector_chunks: list[dict], n_chunks: int) -> list[dict]:
    """Combina resultados de grafo y vector, elimina duplicados y re-rankea.

    Prioridad: chunks que aparecen en ambas fuentes primero,
    luego vector (tiene score), luego graph.
    """
    vector_ids = {c.get("chunk_id") for c in vector_chunks if c.get("chunk_id")}
    graph_ids = {c.get("chunk_id") for c in graph_chunks if c.get("chunk_id")}
    overlap_ids = vector_ids & graph_ids

    seen: set[str] = set()
    result = []

    # 1. Chunks en ambas fuentes (mas relevantes)
    for chunk in vector_chunks:
        cid = chunk.get("chunk_id")
        if cid in overlap_ids and cid not in seen:
            chunk["retrieval_source"] = "graph+vector"
            result.append(chunk)
            seen.add(cid)

    # 2. Solo vector (tienen score de similitud)
    for chunk in vector_chunks:
        cid = chunk.get("chunk_id")
        if cid not in seen:
            result.append(chunk)
            seen.add(cid)

    # 3. Solo graph (sin score pero con contexto relacional)
    for chunk in graph_chunks:
        cid = chunk.get("chunk_id")
        if cid not in seen:
            result.append(chunk)
            seen.add(cid)

    return result[:n_chunks]


# --------------------------------------------------------------------------- #
# Filtros de metadata                                                           #
# --------------------------------------------------------------------------- #

def _infer_filters(question: str) -> dict | None:
    """Usa Claude Haiku para inferir filtros de metadata relevantes para el query."""
    available = get_metadata_values()

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"Valores disponibles en la base de documentos:\n"
        f"{json.dumps(available, ensure_ascii=False, indent=2)}\n\n"
        "Decide si aplicar filtros de metadata para acotar la busqueda. "
        "Reglas:\n"
        "- Solo filtra si la pregunta claramente se refiere a un valor especifico.\n"
        "- Si la pregunta es general o abarca multiples temas, no filtres.\n"
        "- Usa EXACTAMENTE los valores que aparecen en la lista de disponibles.\n"
        "- Puedes filtrar por uno o mas campos a la vez.\n"
        "Responde UNICAMENTE con un objeto JSON valido. "
        'Ejemplos: {"categoria": "Precios"} o {"source": "Procedimiento_X.pdf"} o {}'
    )
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    filters = parse_json_response(raw)
    return filters if filters else None


# --------------------------------------------------------------------------- #
# Construccion del contexto                                                     #
# --------------------------------------------------------------------------- #

def _build_graph_context(question: str) -> str:
    """Extrae relaciones relevantes del grafo y las formatea como texto para el LLM.

    En lugar de solo usar el grafo para seleccionar chunks, envía explícitamente
    las relaciones entre entidades para que Claude pueda razonar sobre ellas.
    """
    matches = graphstore.search_entities_semantic(question, top_k=5, threshold=0.3)
    if not matches:
        return ""

    lines = []
    seen_relations: set[str] = set()

    for match in matches:
        neighbors = graphstore.get_neighbors(match["id"], direction="both")
        for n in neighbors:
            rel = n["rel"]
            if rel.startswith("<-"):
                triple = f"{n['label']} {rel[2:]} {match['label']}"
            else:
                triple = f"{match['label']} {rel} {n['label']}"
            if triple not in seen_relations:
                seen_relations.add(triple)
                lines.append(f"- {triple}")

    if not lines:
        return ""

    return "Relaciones entre entidades (del grafo de conocimiento):\n" + "\n".join(lines)


def _build_context(chunks: list[dict]) -> str:
    """Formatea los chunks recuperados como contexto para Claude."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "desconocido")
        page = chunk.get("page", "")
        score = chunk.get("score")
        retrieval = chunk.get("retrieval_source", "vector")

        page_info = f" - pagina {page}" if page else ""
        score_info = f" | relevancia: {score}" if score is not None else ""
        parts.append(
            f"[Fragmento {i} | {source}{page_info}{score_info} | via: {retrieval}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------- #
# API publica                                                                   #
# --------------------------------------------------------------------------- #

def ask(
        
    question: str,
    filters: dict | None = None,
    n_chunks: int = N_CHUNKS,
) -> dict:
    """Responde una pregunta usando RAG hibrido (grafo + vector).

    Args:
        question: Pregunta del usuario.
        filters:  Filtros de metadata. Si es None se infieren automaticamente.
        n_chunks: Numero maximo de fragmentos a usar como contexto.

    Returns:
        Dict con answer, sources, question, filters y question_type.
    """
    if filters is None:
        filters = _infer_filters(question)

    vector_chunks = _vector_retrieval(question, n_chunks, filters)
    graph_chunks = _graph_retrieval(question, n_chunks)

    chunks = _merge_chunks(graph_chunks, vector_chunks, n_chunks)

    if not chunks:
        return {
            "answer": "No encontre documentacion relevante para responder esta pregunta.",
            "sources": [],
            "question": question,
        }

    context = _build_context(chunks)
    graph_context = _build_graph_context(question)

    user_message = (
        f"{graph_context + chr(10) + chr(10) if graph_context else ''}"
        f"Documentacion disponible:\n\n{context}\n\n"
        f"---\n\nPregunta: {question}"
    )

    response = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return {
        "answer": response.content[0].text,
        "sources": [
            {
                "source": c.get("source"),
                "page": c.get("page"),
                "score": c.get("score"),
                "proceso": c.get("proceso", ""),
                "tema_principal": c.get("tema_principal", ""),
                "retrieval_source": c.get("retrieval_source"),
            }
            for c in chunks
        ],
        "question": question,
        "filters": filters,
    }


def ask_stream(
    question: str,
    filters: dict | None = None,
    n_chunks: int = N_CHUNKS,
):
    """Version streaming de ask(). Hace yield de tokens a medida que llegan."""
    if filters is None:
        filters = _infer_filters(question)

    vector_chunks = _vector_retrieval(question, n_chunks, filters)
    graph_chunks = _graph_retrieval(question, n_chunks)

    chunks = _merge_chunks(graph_chunks, vector_chunks, n_chunks)

    if not chunks:
        yield "No encontre documentacion relevante para responder esta pregunta."
        return

    context = _build_context(chunks)
    graph_context = _build_graph_context(question)

    user_message = (
        f"{graph_context + chr(10) + chr(10) if graph_context else ''}"
        f"Documentacion disponible:\n\n{context}\n\n"
        f"---\n\nPregunta: {question}"
    )

    with _client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text
