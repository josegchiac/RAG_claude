"""Pipeline principal del RAG: indexa documentos y responde preguntas."""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from loader import load_document
from vectorstore import add_chunks, collection_info, delete_by_source
from retriever import ask, ask_stream


DOCS_PATH = Path(__file__).parent.parent / "docs"


# --------------------------------------------------------------------------- #
# Indexacion                                                                    #
# --------------------------------------------------------------------------- #

def index_document(path: str, enrich: bool = True) -> None:
    """Carga, enriquece e indexa un documento."""
    print(f"Cargando: {path}")
    chunks = load_document(path, enrich=enrich)
    print(f"  {len(chunks)} chunks generados")

    inserted = add_chunks(chunks)
    print(f"  {inserted} chunks indexados en ChromaDB")


def index_all(enrich: bool = True) -> None:
    """Indexa todos los documentos PDF y DOCX en la carpeta docs/."""
    files = list(DOCS_PATH.glob("*.pdf")) + list(DOCS_PATH.glob("*.docx"))

    if not files:
        print(f"No se encontraron documentos en {DOCS_PATH}")
        return

    print(f"Encontrados {len(files)} documentos\n")
    for f in files:
        index_document(str(f), enrich=enrich)

    info = collection_info()
    print(f"\nTotal en base vectorial: {info['total_chunks']} chunks")


def reindex_document(path: str, enrich: bool = True) -> None:
    """Elimina e indexa nuevamente un documento (para actualizaciones)."""
    filename = Path(path).name
    delete_by_source(filename)
    print(f"Chunks anteriores de '{filename}' eliminados")
    index_document(path, enrich=enrich)


# --------------------------------------------------------------------------- #
# Consulta                                                                      #
# --------------------------------------------------------------------------- #

def query(question: str, filters: dict | None = None, stream: bool = True) -> None:
    """Imprime la respuesta a una pregunta junto con sus fuentes."""
    print(f"\nPregunta: {question}")
    if filters:
        print(f"Filtros activos: {filters}")
    print("\nRespuesta:")
    print("-" * 60)

    if stream:
        for token in ask_stream(question, filters=filters):
            print(token, end="", flush=True)
        print()
    else:
        result = ask(question, filters=filters)
        if result.get("filters"):
            print(f"Filtros inferidos: {result['filters']}")
        print(result["answer"])
        print("\nFuentes utilizadas:")
        for s in result["sources"]:
            page_info = f" p.{s['page']}" if s["page"] else ""
            print(f"  - {s['source']}{page_info} (relevancia: {s['score']})")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def _print_help() -> None:
    print("""
Uso:
  python main.py index              Indexa todos los docs de /docs
  python main.py index <archivo>    Indexa un archivo especifico
  python main.py reindex <archivo>  Reindexar un archivo actualizado
  python main.py ask "<pregunta>"   Hace una pregunta al RAG
  python main.py info               Muestra info de la base vectorial
""")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "help":
        _print_help()

    elif args[0] == "index":
        if len(args) > 1:
            index_document(args[1])
        else:
            index_all()

    elif args[0] == "reindex" and len(args) > 1:
        reindex_document(args[1])

    elif args[0] == "ask" and len(args) > 1:
        question = args[1]
        # Filtros opcionales como pares clave=valor: python main.py ask "..." categoria=RRHH
        filters = {}
        for arg in args[2:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                filters[k] = v
        query(question, filters=filters or None)

    elif args[0] == "info":
        info = collection_info()
        print(f"Coleccion : {info['collection']}")
        print(f"Chunks    : {info['total_chunks']}")
        print(f"Ruta      : {info['chroma_path']}")

    else:
        _print_help()
