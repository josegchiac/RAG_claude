"""Utilidades compartidas entre modulos."""

import re
import json


def _escape_newlines_in_strings(text: str) -> str:
    """Escapa saltos de linea literales dentro de strings JSON.

    Claude Haiku a veces incluye newlines reales dentro de valores de string
    en lugar de la secuencia de escape \\n, lo que produce JSON invalido.
    Este parser recorre el texto caracter a caracter para corregirlo.
    """
    result = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == "\\" and in_string:
            result.append(char)
            escape_next = True
        elif char == '"':
            in_string = not in_string
            result.append(char)
        elif char == "\n" and in_string:
            result.append("\\n")
        elif char == "\r" and in_string:
            result.append("\\r")
        elif char == "\t" and in_string:
            result.append("\\t")
        else:
            result.append(char)

    return "".join(result)


def parse_json_response(raw: str, expect: type = dict) -> dict | list:
    """Parsea JSON desde una respuesta de Claude de forma tolerante a errores comunes.

    Limpia los problemas mas frecuentes que produce Claude Haiku antes de parsear:
    - Saltos de linea literales dentro de valores de string
    - Trailing commas antes de } o ]
    - Texto fuera del objeto/array JSON

    Args:
        raw:    Texto crudo de la respuesta del LLM.
        expect: tipo esperado (dict o list). Determina que delimitadores buscar.

    Returns:
        dict o list parseado.

    Raises:
        ValueError: si no se encuentra un bloque JSON valido tras la limpieza.
    """
    text = raw.strip()

    # 0. Eliminar bloques de código markdown (```json ... ``` o ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # 1. Escapar newlines literales dentro de strings (causa de "Expecting ',' delimiter")
    text = _escape_newlines_in_strings(text)

    # 2. Eliminar trailing commas antes de } o ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 3. Extraer solo el bloque JSON, ignorando texto extra alrededor
    if expect is list:
        start = text.find("[")
        end = text.rfind("]") + 1
    else:
        start = text.find("{")
        end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No se encontro bloque JSON en la respuesta: {raw[:200]}")

    return json.loads(text[start:end])
