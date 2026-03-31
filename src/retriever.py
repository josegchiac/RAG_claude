"""Conecta la busqueda vectorial con Claude para generar respuestas."""

import json
import anthropic
from vectorstore import search, get_metadata_values


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


def _infer_filters(question: str) -> dict | None:
    """Usa Claude Haiku para inferir filtros de metadata relevantes para el query.

    Consulta los valores disponibles en ChromaDB dinamicamente para que el
    routing funcione aunque se agreguen nuevos documentos o categorias.
    Retorna None si no es necesario aplicar filtros.
    """
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
    start = raw.find("{")
    end = raw.rfind("}") + 1
    filters = json.loads(raw[start:end])
    return filters if filters else None


def _build_context(chunks: list[dict]) -> str:
    """Formatea los chunks recuperados como contexto para Claude."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "desconocido")
        page = chunk.get("page", "")
        score = chunk.get("score", 0)
        page_info = f" - pagina {page}" if page else ""
        parts.append(
            f"[Fragmento {i} | {source}{page_info} | relevancia: {score}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def ask(
    question: str,
    filters: dict | None = None,
    n_chunks: int = N_CHUNKS,
) -> dict:
    """Responde una pregunta usando RAG.

    Args:
        question: Pregunta del usuario.
        filters:  Filtros de metadata para acotar la busqueda. Si es None,
                  se infieren automaticamente desde el query via Claude Haiku.
        n_chunks: Numero de fragmentos a recuperar.

    Returns:
        Dict con:
          - answer:   Respuesta generada por Claude.
          - sources:  Lista de chunks usados como contexto.
          - question: Pregunta original.
          - filters:  Filtros aplicados (inferidos o explícitos).
    """
    # 1. Inferir filtros si no se pasaron explicitamente
    if filters is None:
        filters = _infer_filters(question)

    # 2. Recuperar chunks relevantes
    chunks = search(question, n_results=n_chunks, filters=filters)

    if not chunks:
        return {
            "answer": "No encontre documentacion relevante para responder esta pregunta.",
            "sources": [],
            "question": question,
        }

    # 2. Construir contexto
    context = _build_context(chunks)

    # 3. Llamar a Claude con el contexto
    user_message = (
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
                "categoria": c.get("categoria", ""),
                "proceso": c.get("proceso", ""),
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
    """Version streaming de ask(). Hace yield de tokens a medida que llegan.

    Si filters es None, se infieren automaticamente desde el query via Claude Haiku.

    Uso:
        for token in ask_stream("pregunta"):
            print(token, end="", flush=True)
    """
    if filters is None:
        filters = _infer_filters(question)

    chunks = search(question, n_results=n_chunks, filters=filters)

    if not chunks:
        yield "No encontre documentacion relevante para responder esta pregunta."
        return

    context = _build_context(chunks)
    user_message = (
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
