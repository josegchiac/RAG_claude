# RAG con Claude

Sistema de preguntas y respuestas sobre documentos internos usando ChromaDB como base vectorial y Claude como modelo de lenguaje.

## Requisitos

- Python 3.10+
- Una API key de Anthropic ([console.anthropic.com](https://console.anthropic.com))

## Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd RAG_claude

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API key
cp .env.example .env
# Editar .env y pegar tu ANTHROPIC_API_KEY
```

## Uso

Desde la carpeta `src/`:

```bash
cd src

# Hacer una pregunta (la base vectorial ya viene incluida)
python main.py ask "¿Cuál es el procedimiento para incremento de precios?"

# Ver información de la base vectorial
python main.py info

# Indexar un nuevo documento (requiere tokens de API)
python main.py index ../docs/nuevo_documento.pdf

# Reindexar un documento actualizado
python main.py reindex ../docs/documento.pdf
```

## Estructura

```
RAG_claude/
├── src/
│   ├── main.py          # CLI: index / ask / info
│   ├── loader.py        # Carga PDFs y DOCX, genera chunks
│   ├── vectorstore.py   # ChromaDB: guardar y buscar chunks
│   └── retriever.py     # Conecta búsqueda vectorial con Claude
├── docs/                # Documentos fuente (PDF, DOCX)
├── data/
│   └── chroma_db/       # Base vectorial (incluida, no requiere re-indexar)
├── docs_config.json     # Metadata base por documento
├── requirements.txt
└── .env.example
```

## Agregar documentos nuevos

1. Copiar el PDF o DOCX a la carpeta `docs/`
2. (Opcional) Agregar metadata en `docs_config.json`:
   ```json
   "nombre_archivo.pdf": {
     "tema_principal": "...",
     "categoria": "...",
     "proceso": "...",
     "responsable": "..."
   }
   ```
3. Indexar: `python main.py index ../docs/nombre_archivo.pdf`

> La indexación usa tokens de la API (Claude Haiku) para enriquecer metadata de cada chunk.

## Modelos utilizados

| Tarea | Modelo |
|---|---|
| Respuestas al usuario | claude-sonnet-4-6 |
| Metadata de chunks + filtros | claude-haiku-4-5-20251001 |
| Embeddings | all-MiniLM-L6-v2 (local, sin costo) |
