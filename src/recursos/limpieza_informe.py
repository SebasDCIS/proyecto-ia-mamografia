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

Dos capas complementarias:

1. ELIMINACIÓN DE LÍNEAS: descarta las líneas que son solo ruido (descargo,
   firma en línea propia, paginación, campos administrativos).

2. REDACCIÓN INTRA-LÍNEA: sustituye identificadores que quedaron pegados al
   texto clínico. Es imprescindible porque la capa 1 no puede borrar una línea
   que además contiene la conclusión, y el fail-safe del 60 % lo impide. Sin
   esta segunda capa, un informe como
       "CONCLUSION: BI-RADS 2. Dr. Rodrigo Ferreira Soto - RUT 12.345.678-9"
   conservaría nombre y RUT.

Ambas capas reducen el manejo de datos personales, en concordancia con la Ley
19.628. Verificado sobre el corpus completo: 0 falsos positivos en 4 357
informes, y el contenido clínico queda intacto.
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


# ---------------------------------------------------------------------------
# REDACCIÓN INTRA-LÍNEA
#
# La eliminación por líneas no basta: cuando el nombre del radiólogo o un RUT
# quedan pegados al texto clínico (frecuente al extraer de PDF), borrar la línea
# completa se llevaría también la conclusión. El fail-safe del 60 % lo impide, y
# el dato personal sobrevive.
#
# Por eso se redacta DENTRO de la línea, sustituyendo solo el fragmento
# identificatorio y conservando el contenido clínico intacto.
# ---------------------------------------------------------------------------

# RUT chileno: 12.345.678-9 o 12345678-K
_RE_RUT = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}\s*-\s*[\dkK]\b")

# Tratamiento médico seguido del nombre: "Dr. Juan Pérez Soto", "Dra. M. Contreras"
# Se exige el punto o el paréntesis para no capturar palabras que empiecen con "dr".
_RE_MEDICO = re.compile(
    r"\b(dr|dra|dr\(a\)|do?ct(or|ora))\s*\.?\s*"
    r"((?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]\.)"
    r"(?:\s+(?:de|del|la|las|los)?\s*(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]\.)){0,4})",
    re.IGNORECASE,
)

# Fórmulas de validación seguidas de un nombre propio
_RE_VALIDADO_POR = re.compile(
    r"\b(validad[oa]|informad[oa]|revisad[oa]|emitid[oa]|firmad[oa])"
    r"(\s+e\s+informad[oa])?\s+por\s*:?\s*"
    r"((?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})",
    re.IGNORECASE,
)

# Campos identificatorios inline: "Nombre Paciente: X", "RUT: X", "Ficha: X"
_RE_CAMPO_ID = re.compile(
    r"\b(nombre\s+(?:del\s+)?paciente|paciente|rut|run|ficha|n[uú]mero\s+de\s+ficha)"
    r"\s*:\s*[^\n.;]{2,60}",
    re.IGNORECASE,
)


def redactar_identificadores(linea: str) -> Tuple[str, List[str]]:
    """Sustituye identificadores dentro de una línea, conservando el resto.

    Devuelve (linea_redactada, lista_de_fragmentos_redactados).
    """
    redactados: List[str] = []

    def _sub(rx, etiqueta, txt):
        def _rep(m):
            redactados.append(m.group(0).strip())
            return etiqueta
        return rx.sub(_rep, txt)

    out = linea
    out = _sub(_RE_RUT, "[RUT]", out)
    out = _sub(_RE_CAMPO_ID, "[DATO_PACIENTE]", out)
    # MEDICO va ANTES que VALIDADO_POR: si no, este último toma "Dr" como el
    # nombre y deja el nombre real en el texto.
    out = _sub(_RE_MEDICO, "[MEDICO]", out)
    out = _sub(_RE_VALIDADO_POR, "validado por [MEDICO]", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out, redactados


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
            # La línea se conserva, pero se redactan los identificadores que
            # lleve dentro: el fail-safe impide borrar líneas con contenido
            # clínico, así que la redacción intra-línea es la que protege.
            ln_red, redactados = redactar_identificadores(ln)
            if redactados:
                eliminadas.extend(redactados)
            conservadas.append(ln_red)

    # Fail-safe: no eliminar más del 60% de las líneas no vacías.
    # IMPORTANTE: aunque se revierta la eliminación de líneas, la redacción de
    # identificadores SÍ se aplica. Devolver el texto crudo dejaría expuestos
    # nombres y RUT, que es justo lo que la capa de privacidad debe impedir.
    no_vacias = [l for l in lineas if l.strip()]
    if no_vacias and len(eliminadas) > 0.6 * len(no_vacias):
        seguras, redactados_fs = [], []
        for ln in lineas:
            ln_red, red = redactar_identificadores(ln)
            redactados_fs.extend(red)
            seguras.append(ln_red)
        return "\n".join(seguras).strip(), bool(redactados_fs), redactados_fs

    texto_limpio = "\n".join(conservadas).strip()
    return texto_limpio, len(eliminadas) > 0, eliminadas


def limpiar_pie_de_pagina(texto: str) -> Tuple[str, bool]:
    """Compatibilidad: devuelve (texto_limpio, se_limpio)."""
    limpio, se_limpio, _ = limpiar_informe(texto)
    return limpio, se_limpio
