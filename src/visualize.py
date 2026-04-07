"""Genera una visualizacion HTML interactiva del grafo de conocimiento."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pyvis.network import Network
import graphstore


OUTPUT_PATH = Path(__file__).parent.parent / "graph.html"

COLORS = {
    "rol":          "#4C9BE8",
    "departamento": "#E8844C",
    "proceso":      "#4CE87A",
    "normativa":    "#E8D44C",
    "herramienta":  "#C44CE8",
    "sistema":      "#4CE8D4",
    "canal":        "#E84C6B",
    "segmento":     "#E8A84C",
    "región":       "#84E84C",
    "cliente":      "#4C84E8",
    "producto":     "#E84CA8",
}


def build_html(output: str | None = None) -> str:
    """Genera el archivo HTML del grafo y retorna su ruta."""
    dest = Path(output) if output else OUTPUT_PATH

    info = graphstore.graph_info()
    print(f"Nodos : {info['total_nodes']}")
    print(f"Aristas: {info['total_edges']}")
    print(f"Tipos  : {info['nodes_by_type']}")

    if info["total_nodes"] == 0:
        print("El grafo esta vacio — indexa documentos primero.")
        return ""

    net = Network(
        height="95vh",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="#ffffff",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150, spring_strength=0.05, damping=0.4)

    for node_id, data in graphstore._graph.nodes(data=True):
        entity_type = data.get("type", "desconocido")
        color = COLORS.get(entity_type, "#aaaaaa")
        n_chunks = len(data.get("chunk_ids", []))
        net.add_node(
            node_id,
            label=data.get("label", node_id),
            title=f"<b>{data.get('label', node_id)}</b><br>Tipo: {entity_type}<br>Chunks: {n_chunks}",
            color=color,
            size=12 + (n_chunks * 3),   # nodos con mas chunks aparecen mas grandes
        )

    for src, dst, edge_data in graphstore._graph.edges(data=True):
        for rel in edge_data.get("rels", {}):
            net.add_edge(
                src, dst,
                label=rel,
                title=rel,
                arrows="to",
                color="#888888",
                font={"size": 9, "color": "#cccccc"},
            )

    # Leyenda como nodo flotante
    legend_html = (
        "<div style='position:fixed;bottom:20px;left:20px;background:#2a2a4a;"
        "padding:12px;border-radius:8px;font-family:sans-serif;font-size:12px;color:#fff'>"
        "<b>Tipos de entidad</b><br>"
        + "".join(
            f"<span style='color:{color}'>&#9679;</span> {tipo}<br>"
            for tipo, color in COLORS.items()
        )
        + "</div>"
    )

    net.set_options("""
    {
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.2 },
        "font": { "align": "middle" }
      },
      "physics": {
        "barnesHut": {
          "avoidOverlap": 1.0,
          "springLength": 150,
          "springConstant": 0.05,
          "damping": 0.4,
          "centralGravity": 0.3
        },
        "minVelocity": 1.5,
        "maxVelocity": 30,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "updateInterval": 25
        }
      },
      "nodes": {
        "font": { "size": 13 },
        "margin": 12
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    fix_js = """
<script>
network.on("dragEnd", function(params) {
  params.nodes.forEach(function(nodeId) {
    network.body.nodes[nodeId].options.fixed = {x: true, y: true};
  });
});
</script>
"""

    html = net.generate_html()
    html = html.replace("</body>", f"{fix_js}{legend_html}</body>")
    dest.write_text(html, encoding="utf-8")

    print(f"\nVisualización guardada en: {dest}")
    return str(dest)


if __name__ == "__main__":
    path = build_html()
    if path:
        import subprocess
        subprocess.run(["open", path])
