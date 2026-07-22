"""
predict.py - Orquestador end-to-end del sistema de auditoría mamográfica.

Módulo del MVP del proyecto BME513 (Universidad de Valparaíso).

Une los cuatro módulos del pipeline en una única función pública
`procesar_informe`, respetando el orden lógico:

    full_report
        ↓
    Paso 1: extractor_birads      (regex sobre la conclusión)
        ↓
    Paso 2: verificador_birads_ml (DistilBETO, inmediato tras el regex)
                                  Usa búsqueda híbrida si está activada (default)
        ↓
    Paso 3: extractor_recomendacion (regex + TF-IDF)
        ↓
    Paso 4: cotejo_acr             (con verificación ML integrada)
        ↓
    Paso 5: Consolidar dict final

NOVEDAD v2:
- Nuevo parámetro `usar_buscador_hibrido` (default True)
- Cuando está activo, el Paso 2 usa el buscador híbrido para
  localizar el BI-RADS incluso en informes sin encabezado CONCLUSIÓN
- Detecta alertas de omisión (hallazgos sin BI-RADS asignado)

Uso como librería:

    from src.predict import procesar_informe
    resultado = procesar_informe(full_report="...")

Uso como CLI:

    python -m src.predict                              # ejecuta tests inline
    python -m src.predict --input informe.txt          # procesa TXT
    python -m src.predict --input informe.pdf          # procesa PDF
    python -m src.predict --input informe.txt --id paciente_12345
    python -m src.predict --input informe.txt --no-ml  # sin verificador ML
    python -m src.predict --input informe.txt --sin-buscador  # sin búsqueda híbrida
    python -m src.predict --input informe.txt --output resultado.json

Filosofía Human-on-the-Loop:
- El sistema detecta inconsistencias, no decide la conducta clínica.
- Cada salida incluye trazabilidad completa para auditoría.

Autor: Sebastián Inostroza Hurtado
Fecha: Junio 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from src.cotejo_acr import cotejar_birads_vs_recomendacion
from src.extractor_birads import extraer_birads
from src.extractor_recomendacion import (
    clasificar_recomendacion,
    extraer_texto_recomendacion,
)
from src.verificador_birads_ml import (
    verificar_extraccion_birads,
    verificar_extraccion_birads_con_buscador,
)


# =============================================================================
# FUNCIONES AUXILIARES INTERNAS - LECTURA DE ARCHIVOS
# =============================================================================

def _leer_txt(ruta: str) -> str:
    """Lee un archivo de texto plano."""
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()


def _leer_pdf(ruta: str) -> str:
    """Lee un archivo PDF con texto digital usando pdfplumber.

    No soporta PDFs escaneados (imágenes). Si el resultado es vacío,
    probablemente el PDF requiere OCR (no implementado aquí).
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber no está instalado. Instalar con: pip install pdfplumber"
        )

    texto_paginas = []
    with pdfplumber.open(ruta) as pdf:
        for page in pdf.pages:
            texto = page.extract_text()
            if texto:
                texto_paginas.append(texto)

    texto_completo = "\n".join(texto_paginas)

    if not texto_completo.strip():
        raise ValueError(
            f"El PDF '{ruta}' no contiene texto extraíble. "
            f"Probablemente es un PDF escaneado (imagen) que requeriría OCR. "
            f"Esta versión solo soporta PDFs con texto digital."
        )

    return texto_completo


def _leer_archivo(ruta: str, tipo: Optional[str] = None) -> str:
    """Lee un archivo TXT o PDF y devuelve su contenido como string.

    Args:
        ruta: ruta al archivo.
        tipo: 'txt' | 'pdf' | None (inferir de la extensión).

    Returns:
        Contenido del archivo como string.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el tipo no es soportado o el PDF es escaneado.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

    # Inferir tipo desde la extensión si no se especificó
    if tipo is None:
        ext = os.path.splitext(ruta)[1].lower()
        if ext == ".pdf":
            tipo = "pdf"
        elif ext in (".txt", ".text", ""):
            tipo = "txt"
        else:
            raise ValueError(
                f"No se puede inferir el tipo de archivo desde la extensión "
                f"'{ext}'. Usar --type txt o --type pdf explícitamente."
            )

    if tipo == "pdf":
        return _leer_pdf(ruta)
    elif tipo == "txt":
        return _leer_txt(ruta)
    else:
        raise ValueError(f"Tipo de archivo no soportado: '{tipo}'. Usar 'txt' o 'pdf'.")


# =============================================================================
# FUNCIÓN AUXILIAR INTERNA - CONSOLIDAR RESULTADO
# =============================================================================

def _construir_resultado_consolidado(
    r_birads: Dict[str, Any],
    verificacion_ml: Optional[Dict[str, Any]],
    r_rec: Dict[str, Any],
    resultado_cotejo: Dict[str, Any],
    informe_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Empaqueta los 4 resultados intermedios en el output final consolidado.

    El resultado se organiza por bloques temáticos (uno por módulo) para
    facilitar consumo por dashboards o sistemas downstream.
    """
    # Sub-dict: birads
    bloque_birads = {
        "valor": r_birads.get("birads_conclusion"),
        "confianza": r_birads.get("confianza"),
        "fuente": r_birads.get("fuente"),
        "encabezado_detectado": r_birads.get("encabezado_conclusion"),
        "menciones_adicionales": r_birads.get("menciones_adicionales", []),
    }

    # Sub-dict: verificador ML (Módulo 4). DESACTIVADO por defecto.
    # Se midió por cuatro vías que no aporta sobre la extracción reglada, que es
    # la única autoridad del sistema. Se conserva para reproducir su evaluación
    # (ver docs/BITACORA.md).
    if verificacion_ml is not None:
        bloque_verif = {
            "rol": "verificador_descartado_ver_bitacora",
            "estado": verificacion_ml.get("estado_verificacion"),
            "birads_ml": verificacion_ml.get("birads_predicho_ml"),
            "confianza_ml": verificacion_ml.get("confianza_ml"),
            "coincide_con_regex": verificacion_ml.get("coincide_con_regex"),
            "regla_aplicada": verificacion_ml.get("regla_aplicada"),
            "mensaje": verificacion_ml.get("mensaje"),
            "modo": verificacion_ml.get("modo", "bloque_conclusion"),
        }
        # Si hay alerta de omisión, incluirla
        if "alerta_omision" in verificacion_ml:
            bloque_verif["alerta_omision"] = verificacion_ml["alerta_omision"]
    else:
        bloque_verif = {"rol": "verificador_descartado_ver_bitacora",
                        "estado": "no_ejecutado", "birads_ml": None}

    # Sub-dict: recomendacion
    bloque_rec = {
        "texto_original": r_rec.get("trazabilidad", {}).get("texto_original", ""),
        "texto_normalizado": r_rec.get("trazabilidad", {}).get("texto_normalizado", ""),
        "categoria_principal": r_rec.get("categoria_principal"),
        "categorias_detectadas": r_rec.get("categorias_detectadas", []),
        "confianza": r_rec.get("confianza"),
        "metodo": r_rec.get("metodo"),
        "fuente_extraccion": r_rec.get("fuente_extraccion"),
        "extraccion_dual": r_rec.get("extraccion_dual"),
    }

    # Sub-dict: cotejo_acr
    bloque_cotejo = {
        "estado": resultado_cotejo.get("estado"),
        "alerta": resultado_cotejo.get("requiere_alerta"),
        "severidad": resultado_cotejo.get("severidad"),
        "recomendacion_esperada": resultado_cotejo.get("recomendacion_esperada"),
        "regla_aplicada": resultado_cotejo.get("trazabilidad", {}).get("regla_aplicada"),
        "mensaje": resultado_cotejo.get("mensaje"),
    }

    return {
        "informe_id": informe_id,
        "timestamp": datetime.now().isoformat(),
        "birads": bloque_birads,
        "verificacion_ml": bloque_verif,
        "recomendacion": bloque_rec,
        "cotejo_acr": bloque_cotejo,
        "confiabilidad_tecnica_global": resultado_cotejo.get("confiabilidad_tecnica"),
    }


def _resultado_de_error(
    informe_id: Optional[str],
    paso: str,
    error: Exception,
) -> Dict[str, Any]:
    """Construye un resultado consolidado para errores no recuperables.

    Permite que `procesar_informe` nunca lance excepciones hacia el exterior,
    facilitando el procesamiento batch.
    """
    return {
        "informe_id": informe_id,
        "timestamp": datetime.now().isoformat(),
        "estado_procesamiento": "error",
        "paso_fallido": paso,
        "error_tipo": type(error).__name__,
        "error_mensaje": str(error),
        "birads": {"valor": None, "confianza": None, "fuente": None,
                   "encabezado_detectado": None, "menciones_adicionales": []},
        "verificacion_ml": {"estado": "no_ejecutado", "birads_ml": None},
        "recomendacion": {"texto_original": "", "texto_normalizado": "",
                          "categoria_principal": None, "categorias_detectadas": [],
                          "confianza": None, "metodo": None},
        "cotejo_acr": {"estado": "error", "alerta": False, "severidad": None,
                       "recomendacion_esperada": None, "regla_aplicada": None,
                       "mensaje": f"Procesamiento falló en {paso}: {error}"},
        "confiabilidad_tecnica_global": "no_aplicable",
    }


def _resultado_alerta_omision(
    informe_id: Optional[str],
    verificacion_ml: Dict[str, Any],
) -> Dict[str, Any]:
    """Construye un resultado especial cuando el buscador detecta omisión.

    Este caso ocurre cuando el informe tiene hallazgos radiológicos pero
    NO tiene BI-RADS asignado. Es una alerta clínica de omisión que debe
    escalarse a revisión humana.
    """
    alerta = verificacion_ml.get("alerta_omision", {})

    return {
        "informe_id": informe_id,
        "timestamp": datetime.now().isoformat(),
        "estado_procesamiento": "alerta_omision",
        "birads": {
            "valor": None,
            "confianza": "no_detectado",
            "fuente": "buscador_hibrido",
            "encabezado_detectado": None,
            "menciones_adicionales": [],
        },
        "verificacion_ml": {
            "estado": "alerta_omision_buscador",
            "birads_ml": None,
            "mensaje": verificacion_ml.get("mensaje", ""),
            "alerta_omision": alerta,
        },
        "recomendacion": {
            "texto_original": "",
            "texto_normalizado": "",
            "categoria_principal": None,
            "categorias_detectadas": [],
            "confianza": None,
            "metodo": None,
        },
        "cotejo_acr": {
            "estado": "no_procesable",
            "alerta": True,
            "severidad": alerta.get("severidad", "alta"),
            "recomendacion_esperada": None,
            "regla_aplicada": "alerta_omision_birads",
            "mensaje": alerta.get(
                "mensaje",
                "El informe tiene hallazgos radiológicos pero no tiene "
                "una categoría BI-RADS asignada."
            ),
            "accion_sugerida": alerta.get("accion_sugerida", ""),
        },
        "confiabilidad_tecnica_global": "no_aplicable",
    }


# =============================================================================
# FUNCIÓN PÚBLICA - procesar_informe
# =============================================================================

def procesar_informe(
    full_report: str,
    recommendations_col: Optional[str] = None,
    informe_id: Optional[str] = None,
    usar_verificador_ml: bool = False,
    usar_buscador_hibrido: bool = True,
    usar_ner_recomendacion: bool = False,
) -> Dict[str, Any]:
    """Procesa un informe mamográfico end-to-end a través del pipeline.

    Ejecuta los 4 módulos en orden:
        1. Extractor BI-RADS (regex sobre la conclusión)
        2. Verificador ML (DistilBETO, inmediato tras el regex)
           - Con buscador híbrido si usar_buscador_hibrido=True (default)
           - Con método antiguo (bloque conclusión) si False
        3. Extractor de recomendación (regex + TF-IDF)
        4. Cotejo BI-RADS vs recomendación (con verificación ML integrada)

    Args:
        full_report: texto completo del informe mamográfico.
        recommendations_col: opcional, contenido del campo Recommendations
            si está separado del full_report (más eficiente que re-extraerlo).
        informe_id: identificador opcional del informe para auditoría.
        usar_verificador_ml: si False, omite la verificación ML
            (más rápido, útil para tests o procesamiento batch ligero).
        usar_verificador_ml: desactivado por defecto. El verificador DistilBETO
            se evaluó por cuatro vías y no aporta sobre la vía reglada: la
            ablación muestra que lee el número declarado (0,939 -> 0,544), el
            arbitraje no se activa en ninguno de los 4357 informes, un regex de
            una línea lo empata sobre su propia ventana (99,82% vs 99,78%), y no
            recupera los formatos con typo que sí recupera la tolerancia de
            edición. Ver docs/BITACORA.md. Se conserva el parámetro para
            reproducir la evaluación.
        usar_buscador_hibrido: si True (default), usa la búsqueda híbrida
            para localizar el BI-RADS en informes sin encabezado CONCLUSIÓN
            explícito. Si False, usa el método antiguo (solo bloque conclusión).

    Returns:
        Dict consolidado con la decisión clínica y trazabilidad completa.
        Ver estructura en _construir_resultado_consolidado.

        Casos especiales:
        - Si el buscador detecta omisión (hallazgos sin BI-RADS):
          devuelve un resultado con estado_procesamiento="alerta_omision"
        - Si ocurre error en algún paso:
          devuelve un resultado con estado_procesamiento="error"

        Nunca lanza excepciones hacia el exterior.
    """
    # ========================================================================
    # PASO 0: Limpiar líneas de ruido (descargo, firma, datos del paciente)
    #         antes de procesar. Evita que el NER (y las reglas) confundan el
    #         descargo o la firma con la recomendación, y mejora la privacidad.
    # ========================================================================
    _lineas_limpiadas = []
    try:
        from src.recursos.limpieza_informe import limpiar_informe
        full_report, _pie_recortado, _lineas_limpiadas = limpiar_informe(full_report)
    except Exception:
        _pie_recortado = False

    # ========================================================================
    # PASO 1: Extraer BI-RADS (regex)
    # ========================================================================
    try:
        r_birads = extraer_birads(full_report)
    except Exception as e:
        return _resultado_de_error(informe_id, "extraer_birads", e)

    # ========================================================================
    # PASO 2a: Detección de omisión + reconciliación de confianza (buscador
    #          híbrido, NO requiere ML). Se ejecuta siempre que se use el
    #          buscador, incluso con --no-ml: una omisión es una alerta clínica
    #          y no debe depender de que el ML esté activo.
    # ========================================================================
    resultado_buscador = None
    if usar_buscador_hibrido:
        try:
            from src.buscador_birads import buscar_birads_final
            resultado_buscador = buscar_birads_final(
                full_report, usar_ml_si_ambiguo=False
            )
        except Exception as e:
            print(
                f"[predict] Advertencia: buscador híbrido falló ({e}). "
                f"Continuando sin detección de omisión.",
                file=sys.stderr,
            )
            resultado_buscador = None

        # Omisión: hallazgos/recomendaciones presentes pero sin BI-RADS asignado.
        # Cross-check anti-falso-positivo: solo es omisión si NINGÚN extractor
        # encontró BI-RADS. Si el extractor regex (que tolera typos como "BI-RADS O"
        # por cero, formatos que el buscador puede no capturar) SÍ extrajo un valor,
        # entonces el informe tiene BI-RADS y NO hay omisión.
        if (
            resultado_buscador is not None
            and "alerta" in resultado_buscador
            and r_birads.get("birads_conclusion") is None
        ):
            alerta = resultado_buscador["alerta"]
            verif_omision = {
                "estado_verificacion": "alerta_omision_buscador",
                "regla_aplicada": "buscador_hibrido_omision",
                "mensaje": alerta["mensaje"],
                "alerta_omision": alerta,
                "modo": "busqueda_hibrida",
            }
            return _resultado_alerta_omision(informe_id, verif_omision)

        # Reconciliación de confianza: si el extractor cayó a fallback (informe
        # sin encabezado) pero el buscador localizó el MISMO BI-RADS con mayor
        # confianza (scoring posicional), adoptar la confianza del buscador.
        if (
            resultado_buscador is not None
            and r_birads.get("confianza") in ("baja", "no_detectado")
            and resultado_buscador.get("birads_final") is not None
            and resultado_buscador.get("birads_final") == r_birads.get("birads_conclusion")
            and resultado_buscador.get("confianza") in ("alta", "media")
        ):
            r_birads["confianza"] = resultado_buscador["confianza"]
            r_birads["fuente"] = "buscador_hibrido_posicional"

    # ========================================================================
    # PASO 2: Verificar BI-RADS con ML (inmediato tras el regex)
    # ========================================================================
    if usar_verificador_ml:
        try:
            if usar_buscador_hibrido:
                # NUEVO v2: usar búsqueda híbrida (robusta a informes sin
                # encabezado CONCLUSIÓN, con detección de omisión)
                verificacion_ml = verificar_extraccion_birads_con_buscador(
                    full_report=full_report,
                    birads_regex=r_birads.get("birads_conclusion"),
                    confianza_regex=r_birads.get("confianza", "no_detectado"),
                )

                # Si el buscador detectó omisión de BI-RADS, escalar como
                # resultado especial y terminar (no seguir con módulos 3 y 4)
                if verificacion_ml.get("estado_verificacion") == "alerta_omision_buscador":
                    return _resultado_alerta_omision(informe_id, verificacion_ml)
            else:
                # Método antiguo (bloque conclusión)
                verificacion_ml = verificar_extraccion_birads(
                    full_report=full_report,
                    birads_regex=r_birads.get("birads_conclusion"),
                    confianza_regex=r_birads.get("confianza", "no_detectado"),
                )
        except Exception as e:
            # El ML es un complemento, no crítico. Si falla, continuamos sin él.
            print(
                f"[predict] Advertencia: verificador ML falló ({e}). "
                f"Continuando sin verificación ML.",
                file=sys.stderr,
            )
            verificacion_ml = None
    else:
        verificacion_ml = None

    # ========================================================================
    # PASO 3: Extraer texto y clasificar recomendación
    # ========================================================================
    try:
        # 3a. Extracción por REGLAS (regex), sin NER todavía
        texto_regex = extraer_texto_recomendacion(
            recommendations_col=recommendations_col,
            full_report=full_report,
            usar_ner=False,
        )

        # 3b. Extracción por NER (si se pidió), en paralelo, para poder comparar
        ner_span = ""
        ner_encontro = False
        ner_disponible = False
        if usar_ner_recomendacion and isinstance(full_report, str) and full_report.strip():
            try:
                from src.extractor_ner import extraer_recomendacion_ner, ner_disponible as _nd
                ner_disponible = _nd()
                _ner = extraer_recomendacion_ner(full_report)
                ner_encontro = _ner.get("encontrado", False)
                ner_span = _ner.get("texto", "")
            except Exception:
                ner_disponible = False

        # 3c. Decidir la fuente USADA: reglas primero; si no hallaron, NER
        if texto_regex["encontrado"]:
            texto_info = texto_regex
            fuente_usada = "regex"
        elif ner_encontro:
            from src.extractor_recomendacion import _normalizar_texto as _norm_ext
            norm_ner, typos_ner = _norm_ext(ner_span)
            texto_info = {
                "texto": ner_span, "texto_normalizado": norm_ner,
                "fuente": "ner_distilbeto", "encontrado": True,
                "encabezado_detectado": None, "typos_corregidos": typos_ner,
            }
            fuente_usada = "ner"
        else:
            texto_info = texto_regex
            fuente_usada = "ninguno"

        # 3d. Comparación regex vs NER (para el panel de apoyo de extracción)
        def _norm_cmp(s):
            import re as _re
            return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()
        regex_span = texto_regex.get("texto", "") if texto_regex["encontrado"] else ""
        if texto_regex["encontrado"] and ner_encontro:
            a, b = _norm_cmp(regex_span), _norm_cmp(ner_span)
            concordancia_ext = "concuerdan" if (a in b or b in a or a == b) else "difieren"
        elif texto_regex["encontrado"]:
            concordancia_ext = "solo_regex"
        elif ner_encontro:
            concordancia_ext = "solo_ner"
        else:
            concordancia_ext = "ninguno"

        extraccion_dual = {
            "regex_span": regex_span,
            "regex_encontro": texto_regex["encontrado"],
            "ner_span": ner_span,
            "ner_encontro": ner_encontro,
            "ner_disponible": ner_disponible,
            "ner_solicitado": usar_ner_recomendacion,
            "fuente_usada": fuente_usada,
            "concordancia": concordancia_ext,
        }

        if texto_info["encontrado"]:
            r_rec = clasificar_recomendacion(
                texto_info["texto_normalizado"],
                es_ya_normalizado=True,
            )
            r_rec["trazabilidad"]["texto_original"] = texto_info["texto"]
            r_rec["fuente_extraccion"] = texto_info.get("fuente")
            r_rec["extraccion_dual"] = extraccion_dual
        else:
            r_rec = {
                "categorias_detectadas": [],
                "categoria_principal": None,
                "confianza": "no_clasificada",
                "metodo": None,
                "fuente_extraccion": texto_info.get("fuente"),
                "extraccion_dual": extraccion_dual,
                "trazabilidad": {
                    "texto_original": "",
                    "texto_normalizado": "",
                },
            }
    except Exception as e:
        return _resultado_de_error(informe_id, "extraer_recomendacion", e)

    # ========================================================================
    # PASO 4: Cotejo BI-RADS vs recomendación (con verificación ML)
    # ========================================================================
    try:
        resultado_cotejo = cotejar_birads_vs_recomendacion(
            resultado_birads=r_birads,
            resultado_recomendacion=r_rec,
            verificacion_ml=verificacion_ml,
        )
    except Exception as e:
        return _resultado_de_error(informe_id, "cotejo_acr", e)

    # ========================================================================
    # PASO 5: Consolidar todo en el dict de salida
    # ========================================================================
    return _construir_resultado_consolidado(
        r_birads=r_birads,
        verificacion_ml=verificacion_ml,
        r_rec=r_rec,
        resultado_cotejo=resultado_cotejo,
        informe_id=informe_id,
    )


# =============================================================================
# CLI (Command Line Interface)
# =============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos del CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description=(
            "Orquestador end-to-end del sistema de auditoría mamográfica. "
            "Procesa un informe (TXT o PDF) y devuelve resultado JSON consolidado."
        ),
        epilog=(
            "Ejemplos:\n"
            "  python -m src.predict                                # tests inline\n"
            "  python -m src.predict --input informe.txt            # procesar TXT\n"
            "  python -m src.predict --input informe.pdf            # procesar PDF\n"
            "  python -m src.predict --input informe.txt --no-ml    # sin ML\n"
            "  python -m src.predict --input informe.txt --sin-buscador  # metodo antiguo\n"
            "  python -m src.predict --input informe.txt --output resultado.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Ruta al archivo de entrada (.txt o .pdf). "
             "Si se omite, se ejecutan los tests inline.",
    )
    parser.add_argument(
        "--type", "-t",
        choices=["txt", "pdf"],
        default=None,
        help="Tipo de archivo. Si se omite, se infiere de la extensión.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Identificador opcional del informe (para auditoría).",
    )
    parser.add_argument(
        "--con-ml",
        action="store_true",
        help="Activa el verificador ML del Módulo 4. DESACTIVADO por defecto: "
             "se midió que no aporta sobre la vía reglada (ver docs/BITACORA.md). "
             "Se conserva para reproducir su evaluación.",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help=argparse.SUPPRESS,   # obsoleto: el ML ya viene desactivado
    )
    parser.add_argument(
        "--sin-buscador",
        action="store_true",
        help="Desactiva el buscador híbrido (usa método antiguo de "
             "bloque CONCLUSIÓN).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Guardar el resultado JSON en este archivo. "
             "Si se omite, se imprime en stdout.",
    )
    return parser


def _ejecutar_cli(argv: Optional[list] = None) -> int:
    """Punto de entrada del CLI. Devuelve el código de salida."""
    args = _build_argparser().parse_args(argv)

    # Sin --input: ejecutar los tests inline
    if args.input is None:
        _ejecutar_tests()
        return 0

    # Leer el archivo de entrada
    try:
        full_report = _leer_archivo(args.input, tipo=args.type)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error inesperado al leer el archivo: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # Procesar el informe
    resultado = procesar_informe(
        full_report=full_report,
        informe_id=args.id,
        usar_verificador_ml=args.con_ml,
        usar_buscador_hibrido=not args.sin_buscador,
    )

    # Serializar el resultado a JSON
    salida = json.dumps(resultado, indent=2, ensure_ascii=False, default=str)

    # Guardar a archivo o imprimir en stdout
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(salida)
        print(f"Resultado guardado en: {args.output}")
    else:
        print(salida)

    return 0


# =============================================================================
# TESTS INLINE
# =============================================================================

INFORME_COHERENTE = """
INFORME DE MAMOGRAFIA

HALLAZGOS: Mama densa heterogenea. Sin nodulos sospechosos.
Microcalcificaciones benignas dispersas.

CONCLUSION: BI-RADS 2 - Hallazgos benignos.

RECOMENDACIONES:
- Se sugiere control mamografico anual.
"""

INFORME_ALERTA_CRITICA = """
INFORME DE MAMOGRAFIA

HALLAZGOS: Masa irregular de 15mm con calcificaciones pleomorfas.
Altamente sospechosa de malignidad.

CONCLUSION: BI-RADS 5 - Lesion altamente sospechosa.

RECOMENDACIONES:
- Se sugiere control anual.
"""

INFORME_ALERTA_ALTA = """
INFORME DE MAMOGRAFIA

HALLAZGOS: Microcalcificaciones agrupadas en cuadrante superior externo.
Densidad sospechosa.

CONCLUSION: BI-RADS 4 - Hallazgo sospechoso.

RECOMENDACIONES:
- Controles segun criterio del medico tratante.
"""

INFORME_COHERENTE_EQUIVALENTE = """
INFORME DE MAMOGRAFIA

HALLAZGOS: Densidad asimetrica que requiere evaluacion adicional.

CONCLUSION: BI-RADS 0 - Estudio incompleto.

RECOMENDACIONES:
- Se sugiere correlacion con ecografia mamaria.
"""

INFORME_SIN_ENCABEZADO = """
Examen de mamografia bilateral.

Se observa densidad asimetrica en cuadrante superior externo con
microcalcificaciones agrupadas de aspecto sospechoso.

Se sugiere biopsia estereotactica.

BI-RADS 4
"""

INFORME_OMISION = """
Se explora dirigidamente mama izquierda.

Se visualiza conducto central prominente en region retroareolar. En region
periareolar de cuadrante inferoexterno se observa conducto prominente con
contenido ecogenico, conformando una imagen de aproximadamente 18mm.

Se sugiere estudio histologico de esta imagen para descartar lesiones
papilares.
"""


def _ejecutar_tests() -> None:
    """Suite de tests inline. Ejecutar con: python -m src.predict"""
    print("=" * 75)
    print("TESTS DE src/predict.py")
    print("=" * 75)

    casos = [
        {
            "nombre": "T1: Informe coherente (BI-RADS 2 + control anual)",
            "input": {"full_report": INFORME_COHERENTE, "informe_id": "test_T1"},
            "esperado": {
                "birads.valor": 2,
                "cotejo_acr.estado": "coherente",
                "cotejo_acr.alerta": False,
            },
        },
        {
            "nombre": "T2: Alerta crítica (BI-RADS 5 + control anual)",
            "input": {"full_report": INFORME_ALERTA_CRITICA, "informe_id": "test_T2"},
            "esperado": {
                "birads.valor": 5,
                "cotejo_acr.estado": "incoherente",
                "cotejo_acr.alerta": True,
                "cotejo_acr.severidad": "critica",
            },
        },
        {
            "nombre": "T3: Alerta crítica (BI-RADS 4 + criterio medico, sospecha sin acción diagnóstica)",
            "input": {"full_report": INFORME_ALERTA_ALTA, "informe_id": "test_T3"},
            "esperado": {
                "birads.valor": 4,
                "cotejo_acr.estado": "incoherente",
                "cotejo_acr.alerta": True,
                "cotejo_acr.severidad": "critica",
            },
        },
        {
            "nombre": "T4: Coherente equivalente (BI-RADS 0 + correlacion ecografica)",
            "input": {"full_report": INFORME_COHERENTE_EQUIVALENTE, "informe_id": "test_T4"},
            "esperado": {
                "birads.valor": 0,
                "cotejo_acr.estado": "coherente_equivalente",
                "cotejo_acr.alerta": False,
            },
        },
        {
            "nombre": "T5: Sin verificador ML (modo rapido)",
            "input": {
                "full_report": INFORME_COHERENTE,
                "informe_id": "test_T5",
                "usar_verificador_ml": False,
            },
            "esperado": {
                "verificacion_ml.estado": "no_ejecutado",
                "cotejo_acr.alerta": False,
            },
        },
        {
            "nombre": "T6: Informe vacio (manejo de error)",
            "input": {"full_report": "", "informe_id": "test_T6"},
            "esperado": {
                # Se espera que no lance excepción y devuelva un dict válido
            },
        },
        {
            "nombre": "T7: Informe sin encabezado CONCLUSION (buscador hibrido)",
            "input": {
                "full_report": INFORME_SIN_ENCABEZADO,
                "informe_id": "test_T7",
                "usar_verificador_ml": False,  # Sin ML para test determinista
            },
            "esperado": {
                # El buscador híbrido debe localizar el BI-RADS 4 aunque no haya
                # encabezado CONCLUSIÓN, y el cotejo con la biopsia sugerida
                # debe resultar coherente (sin alerta).
                "birads.valor": 4,
                "cotejo_acr.estado": "coherente",
                "cotejo_acr.alerta": False,
            },
        },
        {
            "nombre": "T8: Alerta de omision (hallazgos sin BI-RADS asignado)",
            "input": {
                "full_report": INFORME_OMISION,
                "informe_id": "test_T8",
                "usar_verificador_ml": False,  # Omisión NO depende del ML
            },
            "esperado": {
                # Hay hallazgos y recomendación (biopsia/histología) pero NINGÚN
                # BI-RADS asignado: el buscador debe disparar la alerta de omisión
                # incluso con el ML desactivado.
                "birads.valor": None,
                "cotejo_acr.alerta": True,
                "cotejo_acr.severidad": "alta",
            },
        },
    ]

    n_pasados = 0

    for caso in casos:
        try:
            resultado = procesar_informe(**caso["input"])
        except Exception as e:
            print(f"  [FALLA] {caso['nombre']}")
            print(f"          Excepción: {type(e).__name__}: {e}")
            continue

        # Verificar que el resultado sea un dict válido
        if not isinstance(resultado, dict):
            print(f"  [FALLA] {caso['nombre']}: resultado no es dict")
            continue

        # Verificar campos esperados (con notación 'bloque.campo')
        checks = []
        for clave, valor_esperado in caso["esperado"].items():
            bloque, campo = clave.split(".")
            valor_real = resultado.get(bloque, {}).get(campo)
            checks.append((
                valor_real == valor_esperado,
                f"{clave}: esperado={valor_esperado}, obtenido={valor_real}",
            ))

        paso = all(ok for ok, _ in checks) if checks else True
        estado_str = "PASA" if paso else "FALLA"

        if paso:
            n_pasados += 1
            print(f"  [{estado_str}] {caso['nombre']}")
        else:
            print(f"  [{estado_str}] {caso['nombre']}")
            for ok, msg in checks:
                if not ok:
                    print(f"         {msg}")

    print(f"\nResumen: {n_pasados}/{len(casos)} tests pasados")

    if n_pasados == len(casos):
        print("Estado: OK — predict.py listo para uso en producción.")
    else:
        print("Estado: FALLA — revisar los casos que no pasaron.")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    sys.exit(_ejecutar_cli())
