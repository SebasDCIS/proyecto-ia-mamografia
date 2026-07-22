"""
Limpieza de líneas de ruido en informes mamográficos (descargos, firmas,
metadatos administrativos y del paciente).

Los informes reales -sobre todo los extraídos de PDF- contienen líneas que NO
son contenido clínico y confunden a los extractores:

    "* El presente resultado debe correlacionarse con el cuadro clínico *"
    "Informe validado e informado por /Dr(a). Rodrigo Ferreira Soto"
    "Nombre Paciente : ...", "RUT : ...", "Page 1 of 1", firma, etc.

Al extraer texto de un PDF el ORDEN puede alterarse (el pie de página puede
aparecer primero). Por eso NO se corta un rango, sino que se eliminan las
LÍNEAS de ruido dondequiera que estén, conservando el contenido clínico.

Beneficio adicional: eliminar nombre, RUT e ID del paciente reduce el manejo
de datos personales (coherente con la Ley 19.628).
"""

import re
import unicodedata
from typing import Tuple, List

_LINEAS_RUIDO = [
    # Descargo / disclaimer legal
    r"presente\s+(informe|resultado|estudio|examen)\s+debe\s+(ser\s+)?correlacionar",
    r"debe\s+(ser\s+)?correlacionars?e?\s+con\s+(el\s+|los\s+)?(cuadro|contexto)\s+clinic",
    r"(no\s+)?(constituye|reemplaza|sustituye)\s+(un\s+)?diagnostic",
    # Firma / validación
    r"informe\s+validado",
    r"informe\s+(revisado|generado|emitido|informado)\s+(e\s+informado\s+)?por",
    r"medico\s+radiolog",
    r"^\s*atentamente\s*,?\s*$",
    r"firma\s+(electronica|digital|del\s+medico)",
    r"\bdr\s*\(a\)\s*\.?",
    # Metadatos administrativos / paginación
    r"^\s*page\s+\d+\s+of\s+\d+",
    r"^\s*pagina\s+\d+\s+de\s+\d+",
    # Datos del paciente (privacidad)
    r"^\s*nombre\s+paciente\s*:",
    r"^\s*rut\s*:",
    r"^\s*id\s*:",
    r"^\s*edad\s*:",
    r"^\s*fecha\s+(examen|informe|de\s+examen|de\s+informe)\s*:",
]

_RE_RUIDO = [re.compile(p) for p in _LINEAS_RUIDO]

# Términos clínicos: una línea que contenga alguno NO es un nombre suelto.
_TERMINOS_CLINICOS = {
    "control", "biopsia", "ecografia", "ecografico", "mamografia", "mamografico",
    "birads", "acr", "resonancia", "estudio", "examen", "recomendacion",
    "recomienda", "sugiere", "correlacion", "seguimiento", "hallazgos",
    "impresion", "conclusion", "mama", "mamaria", "anual", "meses", "derivacion",
    "complementar", "ultrasonido", "densidad", "mantener", "puncion", "nodulo",
    "quiste", "lesion", "benigno", "maligno", "axila", "parenquima",
}


def _es_nombre_suelto(linea: str) -> bool:
    """Heurística conservadora: True si la línea es solo un nombre propio
    (firma del médico), no contenido clínico."""
    ln = linea.strip()
    if not ln or len(ln) > 45:
        return False
    if re.search(r"[\d:,]", ln):
        return False
    palabras = ln.split()
    if not (2 <= len(palabras) <= 4):
        return False
    if not all(p[0].isupper() for p in palabras if p):
        return False
    ln_norm = _normaliza_linea(ln)
    if any(p in _TERMINOS_CLINICOS for p in ln_norm.split()):
        return False
    return True


def _normaliza_linea(linea: str) -> str:
    t = unicodedata.normalize("NFKD", linea)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def _es_linea_ruido(linea: str) -> bool:
    if not linea.strip():
        return False
    if _es_nombre_suelto(linea):
        return True
    ln = _normaliza_linea(linea).strip().strip("*").strip()
    return any(rx.search(ln) for rx in _RE_RUIDO)


def limpiar_informe(texto: str) -> Tuple[str, bool, List[str]]:
    """Elimina las líneas de ruido del informe, dondequiera que estén."""
    if not isinstance(texto, str) or not texto.strip():
        return texto, False, []

    lineas = texto.split("\n")
    conservadas: List[str] = []
    eliminadas: List[str] = []
    for ln in lineas:
        if _es_linea_ruido(ln):
            eliminadas.append(ln.strip())
        else:
            conservadas.append(ln)

    # Fail-safe: no eliminar más del 60% de las líneas no vacías.
    no_vacias = [l for l in lineas if l.strip()]
    if no_vacias and len(eliminadas) > 0.6 * len(no_vacias):
        return texto, False, []

    texto_limpio = "\n".join(conservadas).strip()
    return texto_limpio, len(eliminadas) > 0, eliminadas


def limpiar_pie_de_pagina(texto: str) -> Tuple[str, bool]:
    """Compatibilidad: devuelve (texto_limpio, se_limpio)."""
    limpio, se_limpio, _ = limpiar_informe(texto)
    return limpio, se_limpio
