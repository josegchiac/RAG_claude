"""Gestiona el grafo de conocimiento con NetworkX.

Cada nodo representa una entidad (rol, departamento, proceso, etc.) y
guarda referencias a los chunk_ids de ChromaDB donde aparece.
Cada arista representa una relacion direccional entre dos entidades.
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path


GRAPH_PATH = Path(__file__).parent.parent / "data" / "knowledge_graph.json"

_graph: nx.DiGraph = nx.DiGraph()


# --------------------------------------------------------------------------- #
# Persistencia                                                                  #
# --------------------------------------------------------------------------- #

def load_graph() -> None:
    """Carga el grafo desde disco si existe.

    Detecta automaticamente si el JSON usa la clave 'links' (NetworkX < 3.4)
    o 'edges' (NetworkX >= 3.4) para mantener compatibilidad entre versiones.
    """
    if GRAPH_PATH.exists():
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        global _graph
        edges_key = "links" if "links" in data else "edges"
        try:
            _graph = nx.node_link_graph(data, directed=True, multigraph=False, edges=edges_key)
        except TypeError:
            # NetworkX < 3.4 no acepta el parametro edges
            _graph = nx.node_link_graph(data, directed=True, multigraph=False)


def save_graph() -> None:
    """Persiste el grafo en disco usando siempre la clave 'links'.

    Fija edges='links' para que el archivo sea legible tanto en NetworkX < 3.4
    como en >= 3.4 (que acepta el parametro edges en node_link_graph).
    """
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = nx.node_link_data(_graph, edges="links")
    except TypeError:
        # NetworkX < 3.4 no acepta el parametro edges (ya usa 'links' por defecto)
        data = nx.node_link_data(_graph)
    GRAPH_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Cargar al importar el modulo (igual que ChromaDB en vectorstore.py)
load_graph()


# --------------------------------------------------------------------------- #
# Escritura                                                                     #
# --------------------------------------------------------------------------- #

def add_chunk_graph(chunk_id: str, entities: list[dict], relations: list[dict]) -> None:
    """Agrega entidades y relaciones de un chunk al grafo.

    Si una entidad ya existe, acumula el chunk_id en su lista.
    Si una relacion ya existe, acumula el chunk_id en su lista.

    Args:
        chunk_id:  ID del chunk en ChromaDB (ej: "Procedimiento_p2_c0").
        entities:  Lista de dicts con id, label, type.
        relations: Lista de dicts con from, rel, to.
    """
    for entity in entities:
        eid = entity["id"]
        if _graph.has_node(eid):
            # Acumular chunk_id sin duplicados
            existing = set(_graph.nodes[eid].get("chunk_ids", []))
            existing.add(chunk_id)
            _graph.nodes[eid]["chunk_ids"] = list(existing)
        else:
            _graph.add_node(
                eid,
                label=entity.get("label", eid),
                type=entity.get("type", "desconocido"),
                chunk_ids=[chunk_id],
            )

    for relation in relations:
        src, rel, dst = relation["from"], relation["rel"], relation["to"]
        # Solo agregar si ambos nodos existen
        if not _graph.has_node(src) or not _graph.has_node(dst):
            continue
        if _graph.has_edge(src, dst):
            existing_rels = _graph[src][dst].get("rels", {})
            chunks_for_rel = set(existing_rels.get(rel, []))
            chunks_for_rel.add(chunk_id)
            existing_rels[rel] = list(chunks_for_rel)
            _graph[src][dst]["rels"] = existing_rels
        else:
            _graph.add_edge(src, dst, rels={rel: [chunk_id]})

    save_graph()


def delete_by_source(source: str) -> None:
    """Elimina del grafo todos los chunk_ids asociados a un documento.

    Si un nodo queda sin chunk_ids, se elimina del grafo.
    """
    nodes_to_remove = []
    for node_id, data in _graph.nodes(data=True):
        updated = [c for c in data.get("chunk_ids", []) if not c.startswith(Path(source).stem)]
        if not updated:
            nodes_to_remove.append(node_id)
        else:
            _graph.nodes[node_id]["chunk_ids"] = updated

    _graph.remove_nodes_from(nodes_to_remove)
    save_graph()


# --------------------------------------------------------------------------- #
# Consulta                                                                      #
# --------------------------------------------------------------------------- #

def get_chunks_for_entity(entity_id: str) -> list[str]:
    """Retorna los chunk_ids de ChromaDB donde aparece una entidad."""
    if not _graph.has_node(entity_id):
        return []
    return _graph.nodes[entity_id].get("chunk_ids", [])


def get_neighbors(
    entity_id: str,
    rel_type: str | None = None,
    direction: str = "out",
) -> list[dict]:
    """Retorna los vecinos de una entidad en el grafo.

    Args:
        entity_id: ID de la entidad origen.
        rel_type:  Si se especifica, filtra por tipo de relacion.
        direction: "out" (entidades a las que apunta), "in" (las que apuntan a esta),
                   "both" (ambas).

    Returns:
        Lista de dicts con id, label, type, rel (relacion), chunk_ids.
    """
    if not _graph.has_node(entity_id):
        return []

    results = []

    if direction in ("out", "both"):
        for _, dst, edge_data in _graph.out_edges(entity_id, data=True):
            for rel, chunks in edge_data.get("rels", {}).items():
                if rel_type and rel != rel_type:
                    continue
                dst_data = _graph.nodes[dst]
                results.append({
                    "id": dst,
                    "label": dst_data.get("label", dst),
                    "type": dst_data.get("type", "desconocido"),
                    "rel": rel,
                    "chunk_ids": chunks,
                })

    if direction in ("in", "both"):
        for src, _, edge_data in _graph.in_edges(entity_id, data=True):
            for rel, chunks in edge_data.get("rels", {}).items():
                if rel_type and rel != rel_type:
                    continue
                src_data = _graph.nodes[src]
                results.append({
                    "id": src,
                    "label": src_data.get("label", src),
                    "type": src_data.get("type", "desconocido"),
                    "rel": f"<-{rel}",
                    "chunk_ids": chunks,
                })

    return results


def find_entities_by_type(entity_type: str) -> list[dict]:
    """Retorna todas las entidades de un tipo dado (ej: 'rol', 'proceso')."""
    return [
        {"id": nid, **data}
        for nid, data in _graph.nodes(data=True)
        if data.get("type") == entity_type
    ]


def search_entities(query: str) -> list[dict]:
    """Busca entidades cuyo id o label contenga el texto dado (case-insensitive)."""
    q = query.lower()
    return [
        {"id": nid, **data}
        for nid, data in _graph.nodes(data=True)
        if q in nid.lower() or q in data.get("label", "").lower()
    ]


def search_entities_semantic(query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
    """Busca entidades por similitud semantica entre el query y los labels del grafo.

    Usa el mismo modelo de embeddings que vectorstore.py (all-MiniLM-L6-v2).
    Como los vectores estan normalizados, el dot product equivale a cosine similarity.

    Args:
        query:     Termino a buscar semanticamente.
        top_k:     Maximo de resultados a retornar.
        threshold: Score minimo de similitud para incluir un resultado (0-1).

    Returns:
        Lista de entidades ordenadas por similitud descendente, con campo "score".
    """
    from vectorstore import embed

    nodes = list(_graph.nodes(data=True))
    if not nodes:
        return []

    labels = [data.get("label", nid) for nid, data in nodes]

    # Embeddear query y todos los labels en un solo batch
    all_embs = np.array(embed([query] + labels))
    query_emb = all_embs[0]
    label_embs = all_embs[1:]

    # Dot product = cosine similarity (vectores normalizados)
    scores = (label_embs @ query_emb).tolist()

    ranked = sorted(zip(scores, nodes), reverse=True)
    return [
        {"id": nid, "score": round(score, 4), **data}
        for score, (nid, data) in ranked[:top_k]
        if score >= threshold
    ]


def get_subgraph_chunk_ids(entity_id: str, depth: int = 2) -> list[str]:
    """Recupera todos los chunk_ids del subgrafo a N niveles de profundidad.

    Util para preguntas relacionales: reune todos los chunks vinculados
    a una entidad y sus vecinos directos/indirectos.
    """
    if not _graph.has_node(entity_id):
        return []

    visited = set()
    frontier = {entity_id}

    for _ in range(depth):
        next_frontier = set()
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            for neighbor in list(_graph.successors(nid)) + list(_graph.predecessors(nid)):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
        frontier = next_frontier

    visited.add(entity_id)
    chunk_ids = []
    for nid in visited:
        chunk_ids.extend(_graph.nodes[nid].get("chunk_ids", []))

    return list(set(chunk_ids))


# --------------------------------------------------------------------------- #
# Utilidades                                                                    #
# --------------------------------------------------------------------------- #

def graph_info() -> dict:
    """Retorna estadisticas basicas del grafo."""
    type_counts: dict[str, int] = {}
    for _, data in _graph.nodes(data=True):
        t = data.get("type", "desconocido")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_nodes": _graph.number_of_nodes(),
        "total_edges": _graph.number_of_edges(),
        "nodes_by_type": type_counts,
        "graph_path": str(GRAPH_PATH),
    }
