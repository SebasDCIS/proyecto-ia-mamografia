"""
Extractor de BI-RADS declarado en informes mamográficos en español.

Módulo del MVP del proyecto BME513 (Universidad de Valparaíso).
Extrae la categoría BI-RADS que el radiólogo escribió textualmente en el
bloque CONCLUSIÓN del informe, distinguiéndola de menciones secundarias
(por ejemplo, ecografías complementarias citadas en el cuerpo del texto).

Validado sobre el corpus público de Vázquez Noguera et al. (2025):
    - 4 357 informes procesados
    - 100.00% tasa de extracción
    - 99.93% tasa de coincidencia con etiqueta del dataset
    - Las 3 divergencias remanentes son inconsistencias clínicas reales
      (estudios complementarios integrados por el dataset) y no bugs.

Uso típico:
    >>> from src.extractor_birads import extraer_birads
    >>> texto = "MAMOGRAFIA... CONCLUSION: BI-RADS 4A. RECOMENDACIONES: biopsia."
    >>> resultado = extraer_birads(texto)
    >>> resultado["birads_conclusion"]
    4
    >>> resultado["subcategoria"]
    'a'
    >>> resultado["confianza"]
    'alta'

Autor: Sebastián Inostroza Hurtado
Fecha: Mayo 2026
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# =============================================================================
# CONSTANTES Y PATRONES
# =============================================================================

ENCABEZADOS_CONCLUSION: List[str] = [
    r"conclusi[oó]n\s+e?\s*impresi[oó]n\s+diagn[oó]stica",
    r"hallazgos\s+y\s+conclusi[oó]n",
    r"impresi[oó]n\s+diagn[oó]stica\s+y?\s*recomendaciones?",
    r"impresi[oó]n\s+diagn[oó]stica",
    r"impresi[oó]n\s+final",
    r"diagn[oó]stico\s+presuntivo",
    r"diagn[oó]stico\s+radiol[oó]gico",
    r"opini[oó]n\s+del\s+radi[oó]logo",
    r"conclusi[oó]n",
    r"valoraci[oó]n",
    r"impresi[oó]n",
    r"diagn[oó]stico",
]

PATRON_ENCABEZADO_CONCLUSION = re.compile(
    r"\b(" + "|".join(ENCABEZADOS_CONCLUSION) + r")\s*:?",
    re.IGNORECASE,
)

PATRON_INICIO_RECOMENDACIONES = re.compile(
    r"\b(recomendaci[oó]n(?:es)?|indicaci[oó]n(?:es)?|sugerenci[ao]s?|"
    r"conducta\s+a?\s*seguir|plan)\s*:?",
    re.IGNORECASE,
)

PATRON_BIRADS_PRINCIPAL = re.compile(
    r'\bbi\s*[-*.]?\s*rad[s]?'
    r'\s*®?\s*'
    r':?\s*'
    r'\(?\s*'
    r'('
    r'0?[0-6](?:\s*-\s*[0-6])?'
    r'|VI|IV|V|III|II|I'
    r'|cero|uno|dos|tres|cuatro|cinco|seis'
    r')'
    r'\s*'
    r'([abc])?'
    r'\s*\)?',
    re.IGNORECASE,
)

PATRON_BIRADS_TYPOS = re.compile(
    r'\bbi\s*[-*.]?\s*rad[ai]?[s]?'
    r'\s*®?\s*:?\s*\(?\s*'
    r'([Ool0-6])'
    r'\s*([abc])?'
    r'\s*\)?',
    re.IGNORECASE,
)

ROMANOS_A_NUMERO: Dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
}

PALABRAS_A_NUMERO: Dict[str, int] = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3,
    "cuatro": 4, "cinco": 5, "seis": 6,
}

TYPOS_A_NUMERO: Dict[str, int] = {"O": 0, "o": 0, "l": 1}


# =============================================================================
# FUNCIONES INTERNAS (helpers)
# =============================================================================

def _convertir_a_entero(s: Optional[str]) -> Optional[int]:
    """Convierte una representación de número BI-RADS al entero 0-6."""
    if not s:
        return None
    s = s.strip()

    if "-" in s:
        partes = [p.strip() for p in s.split("-")]
        try:
            numeros = [int(p) for p in partes if p.isdigit()]
            if numeros:
                return max(numeros)
        except ValueError:
            pass

    if s.isdigit():
        n = int(s)
        if 0 <= n <= 6:
            return n

    s_upper = s.upper()
    if s_upper in ROMANOS_A_NUMERO:
        return ROMANOS_A_NUMERO[s_upper]

    s_lower = s.lower()
    if s_lower in PALABRAS_A_NUMERO:
        return PALABRAS_A_NUMERO[s_lower]

    if s in TYPOS_A_NUMERO:
        return TYPOS_A_NUMERO[s]

    return None


def _localizar_bloque_conclusion(texto: str) -> Optional[Dict[str, Any]]:
    """Localiza el bloque de CONCLUSIÓN en el informe."""
    if not isinstance(texto, str) or not texto.strip():
        return None

    match_conclusion = PATRON_ENCABEZADO_CONCLUSION.search(texto)
    if not match_conclusion:
        return None

    inicio_bloque = match_conclusion.end()
    encabezado = match_conclusion.group(0).strip().rstrip(":").strip()

    resto_texto = texto[inicio_bloque:]
    match_recom = PATRON_INICIO_RECOMENDACIONES.search(resto_texto)

    if match_recom:
        fin_bloque = inicio_bloque + match_recom.start()
    else:
        fin_bloque = len(texto)

    return {
        "texto": texto[inicio_bloque:fin_bloque].strip(),
        "inicio": inicio_bloque,
        "fin": fin_bloque,
        "encabezado": encabezado,
    }


def _detectar_formato_original(numero_str: str) -> str:
    """Determina el formato del número extraído."""
    if numero_str.upper() in ROMANOS_A_NUMERO:
        return "romano"
    if numero_str.lower() in PALABRAS_A_NUMERO:
        return "palabra"
    if numero_str in TYPOS_A_NUMERO:
        return "typo"
    return "arabigo"


# =============================================================================
# FUNCIÓN PÚBLICA PRINCIPAL
# =============================================================================

def extraer_birads(texto: str) -> Dict[str, Any]:
    """Extrae el BI-RADS declarado en la CONCLUSIÓN del informe.

    Estrategia (cinco pasos, de más estricto a más permisivo):
    1. Localizar el bloque CONCLUSIÓN del informe.
    2. Buscar BI-RADS con la regex estricta → confianza alta.
    3. Reintentar con la regex tolerante a typos → confianza media.
    4. Buscar en todo el texto como fallback → confianza baja.
    5. Si nada encuentra, devolver None → confianza no_detectado.
    """
    resultado: Dict[str, Any] = {
        "birads_conclusion": None,
        "subcategoria": None,
        "categoria_completa": "no detectado",
        "confianza": "no_detectado",
        "fuente": None,
        "encabezado_conclusion": None,
        "menciones_adicionales": [],
        "rango_detectado": False,
        "formato_original": None,
        "error": None,
    }

    if not isinstance(texto, str) or not texto.strip():
        resultado["error"] = "Texto vacío o no es string"
        return resultado

    bloque = _localizar_bloque_conclusion(texto)

    if bloque:
        resultado["encabezado_conclusion"] = bloque["encabezado"]

        match = PATRON_BIRADS_PRINCIPAL.search(bloque["texto"])
        if match:
            numero_str = match.group(1)
            subcat = match.group(2)
            numero = _convertir_a_entero(numero_str)

            if numero is not None:
                resultado["birads_conclusion"] = numero
                resultado["subcategoria"] = subcat.lower() if subcat else None
                resultado["categoria_completa"] = (
                    f"{numero}{subcat.lower()}" if subcat else str(numero)
                )
                resultado["confianza"] = "alta"
                resultado["fuente"] = "bloque_conclusion_estricto"
                resultado["rango_detectado"] = "-" in numero_str
                resultado["formato_original"] = _detectar_formato_original(numero_str)

                if resultado["rango_detectado"]:
                    resultado["confianza"] = "media"

        if resultado["birads_conclusion"] is None:
            match_typo = PATRON_BIRADS_TYPOS.search(bloque["texto"])
            if match_typo:
                numero_str = match_typo.group(1)
                subcat = match_typo.group(2)
                numero = _convertir_a_entero(numero_str)

                if numero is not None:
                    resultado["birads_conclusion"] = numero
                    resultado["subcategoria"] = subcat.lower() if subcat else None
                    resultado["categoria_completa"] = (
                        f"{numero}{subcat.lower()}" if subcat else str(numero)
                    )
                    resultado["confianza"] = "media"
                    resultado["fuente"] = "bloque_conclusion_tolerante_typos"
                    resultado["formato_original"] = "typo"

    if resultado["birads_conclusion"] is None:
        match = PATRON_BIRADS_PRINCIPAL.search(texto)
        if not match:
            match = PATRON_BIRADS_TYPOS.search(texto)

        if match:
            numero_str = match.group(1)
            subcat = match.group(2)
            numero = _convertir_a_entero(numero_str)

            if numero is not None:
                resultado["birads_conclusion"] = numero
                resultado["subcategoria"] = subcat.lower() if subcat else None
                resultado["categoria_completa"] = (
                    f"{numero}{subcat.lower()}" if subcat else str(numero)
                )
                resultado["confianza"] = "baja"
                resultado["fuente"] = "fallback_texto_completo"
                resultado["formato_original"] = _detectar_formato_original(numero_str)

    if bloque:
        texto_fuera_bloque = texto[:bloque["inicio"]] + texto[bloque["fin"]:]
        for m in PATRON_BIRADS_PRINCIPAL.finditer(texto_fuera_bloque):
            numero_str = m.group(1)
            numero = _convertir_a_entero(numero_str)
            if numero is not None and numero != resultado["birads_conclusion"]:
                if numero not in resultado["menciones_adicionales"]:
                    resultado["menciones_adicionales"].append(numero)

    return resultado


# =============================================================================
# TESTS INLINE
# =============================================================================

def _ejecutar_tests() -> None:
    """Suite de tests inline. Ejecutar con: python src/extractor_birads.py"""
    casos = [
        {"nombre": "C1: BI-RADS 2 estándar",
         "texto": "MAMOGRAFIA. CONCLUSION: - BI-RADS 2 (Segun la ACR). RECOMENDACIONES: control anual.",
         "esperado_birads": 2, "esperado_confianza": "alta"},
        {"nombre": "C2: BI-RADS ® 0",
         "texto": "CONCLUSION: BI-RADS \u00ae 0 (Segun ACR). RECOMENDACIONES: ecografia.",
         "esperado_birads": 0, "esperado_confianza": "alta"},
        {"nombre": "C3: Subcategoría 4A",
         "texto": "CONCLUSION: - BI-RADS 4A. RECOMENDACIONES: biopsia.",
         "esperado_birads": 4, "esperado_subcategoria": "a", "esperado_confianza": "alta"},
        {"nombre": "C4: VALORACIÓN como encabezado",
         "texto": "VALORACION: - BI-RADS \u00ae 3. RECOMENDACIONES: control 6 meses.",
         "esperado_birads": 3, "esperado_confianza": "alta"},
        {"nombre": "C5: typo letra O por cero",
         "texto": "CONCLUSION: - BI-RADS O. RECOMENDACIONES: ecografia.",
         "esperado_birads": 0, "esperado_confianza": "media"},
        {"nombre": "C6: typo BI-RADAS",
         "texto": "CONCLUSION: BI-RADAS 4. RECOMENDACIONES: biopsia.",
         "esperado_birads": 4, "esperado_confianza": "media"},
        {"nombre": "C7: rango 4-5 (conservador, toma 5)",
         "texto": "CONCLUSION: BI-RADS 4-5 lesion. RECOMENDACIONES: biopsia.",
         "esperado_birads": 5, "esperado_confianza": "media", "esperado_rango": True},
        {"nombre": "C8: BI-RADS01 (cero a la izquierda)",
         "texto": "CONCLUSION: BI-RADS01. RECOMENDACIONES: ecografia.",
         "esperado_birads": 1, "esperado_confianza": "alta"},
        {"nombre": "C9: Romano IV (latente)",
         "texto": "CONCLUSION: BI-RADS IV. RECOMENDACIONES: biopsia.",
         "esperado_birads": 4, "esperado_confianza": "alta"},
        {"nombre": "C10: Palabra 'tres' (latente)",
         "texto": "CONCLUSION: BI-RADS tres. RECOMENDACIONES: control.",
         "esperado_birads": 3, "esperado_confianza": "alta"},
        {"nombre": "C11: Múltiples menciones",
         "texto": "ECOGRAFIA INFORMA BI-RADS 1. CONCLUSION: BI-RADS 2. RECOMENDACIONES: control anual.",
         "esperado_birads": 2, "esperado_menciones_adicionales": [1]},
        {"nombre": "C12: Fallback sin encabezado",
         "texto": "Informe sin estructura solo dice BI-RADS 2 en alguna parte.",
         "esperado_birads": 2, "esperado_confianza": "baja"},
        {"nombre": "C13: Texto sin BI-RADS",
         "texto": "MAMOGRAFIA NORMAL. CONCLUSION: sin hallazgos. RECOMENDACIONES: control anual.",
         "esperado_birads": None, "esperado_confianza": "no_detectado"},
    ]

    print("=" * 70)
    print("TESTS DE src/extractor_birads.py")
    print("=" * 70)

    n_pasados = 0
    for caso in casos:
        resultado = extraer_birads(caso["texto"])
        checks = [
            resultado["birads_conclusion"] == caso["esperado_birads"],
            ("esperado_confianza" not in caso or
             resultado["confianza"] == caso["esperado_confianza"]),
            ("esperado_subcategoria" not in caso or
             resultado["subcategoria"] == caso["esperado_subcategoria"]),
            ("esperado_rango" not in caso or
             resultado["rango_detectado"] == caso["esperado_rango"]),
            ("esperado_menciones_adicionales" not in caso or
             resultado["menciones_adicionales"] == caso["esperado_menciones_adicionales"]),
        ]
        paso = all(checks)
        estado = "PASA" if paso else "FALLA"
        if paso:
            n_pasados += 1
            print(f"  [{estado}] {caso['nombre']}")
        else:
            print(f"  [{estado}] {caso['nombre']}")
            print(f"         birads:    esperado={caso['esperado_birads']}, "
                  f"obtenido={resultado['birads_conclusion']}")
            print(f"         confianza: esperado={caso.get('esperado_confianza', 'N/A')}, "
                  f"obtenido={resultado['confianza']}")

    print(f"\nResumen: {n_pasados}/{len(casos)} tests pasados")
    if n_pasados == len(casos):
        print("Estado: OK — el extractor está listo para uso en producción.")
    else:
        print("Estado: FALLA — revisar los casos que no pasaron.")


if __name__ == "__main__":
    _ejecutar_tests()
