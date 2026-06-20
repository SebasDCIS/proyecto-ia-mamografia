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
        ↓
    Paso 3: extractor_recomendacion (regex + TF-IDF)
        ↓
    Paso 4: cotejo_acr             (con verificación ML integrada)
        ↓
    Paso 5: Consolidar dict final

Uso como librería:

    from src.predict import procesar_informe
    resultado = procesar_informe(full_report="...")

Uso como CLI:

    python -m src.predict                              # ejecuta tests inline
    python -m src.predict --input informe.txt          # procesa TXT
    python -m src.predict --input informe.pdf          # procesa PDF
    python -m src.predict --input informe.txt --id paciente_12345
    python -m src.predict --input informe.txt --no-ml  # sin verificador ML
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
from src.verificador_birads_ml import verificar_extraccion_birads


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

    # Sub-dict: verificacion_ml (puede ser None si se desactivó)
    if verificacion_ml is not None:
        bloque_verif = {
            "estado": verificacion_ml.get("estado_verificacion"),
            "birads_ml": verificacion_ml.get("birads_predicho_ml"),
            "confianza_ml": verificacion_ml.get("confianza_ml"),
            "coincide_con_regex": verificacion_ml.get("coincide_con_regex"),
            "regla_aplicada": verificacion_ml.get("regla_aplicada"),
            "mensaje": verificacion_ml.get("mensaje"),
        }
    else:
        bloque_verif = {"estado": "no_ejecutado", "birads_ml": None}

    # Sub-dict: recomendacion
    bloque_rec = {
        "texto_original": r_rec.get("trazabilidad", {}).get("texto_original", ""),
        "texto_normalizado": r_rec.get("trazabilidad", {}).get("texto_normalizado", ""),
        "categoria_principal": r_rec.get("categoria_principal"),
        "categorias_detectadas": r_rec.get("categorias_detectadas", []),
        "confianza": r_rec.get("confianza"),
        "metodo": r_rec.get("metodo"),
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


# =============================================================================
# FUNCIÓN PÚBLICA - procesar_informe
# =============================================================================

def procesar_informe(
    full_report: str,
    recommendations_col: Optional[str] = None,
    informe_id: Optional[str] = None,
    usar_verificador_ml: bool = True,
) -> Dict[str, Any]:
    """Procesa un informe mamográfico end-to-end a través del pipeline.

    Ejecuta los 4 módulos en orden:
        1. Extractor BI-RADS (regex sobre la conclusión)
        2. Verificador ML (DistilBETO, inmediato tras el regex)
        3. Extractor de recomendación (regex + TF-IDF)
        4. Cotejo BI-RADS vs recomendación (con verificación ML integrada)

    Args:
        full_report: texto completo del informe mamográfico.
        recommendations_col: opcional, contenido del campo Recommendations
            si está separado del full_report (más eficiente que re-extraerlo).
        informe_id: identificador opcional del informe para auditoría.
        usar_verificador_ml: si False, omite la verificación ML
            (más rápido, útil para tests o procesamiento batch ligero).

    Returns:
        Dict consolidado con la decisión clínica y trazabilidad completa.
        Estructura:
            {
                "informe_id": str | None,
                "timestamp": str (ISO),
                "birads": {valor, confianza, fuente, ...},
                "verificacion_ml": {estado, birads_ml, ...},
                "recomendacion": {texto_original, categoria_principal, ...},
                "cotejo_acr": {estado, alerta, severidad, mensaje, ...},
                "confiabilidad_tecnica_global": str,
            }

        En caso de error en cualquier módulo, devuelve un dict con la clave
        `estado_procesamiento: "error"` y los demás campos vacíos. Nunca lanza
        excepciones hacia el exterior.
    """
    # ========================================================================
    # PASO 1: Extraer BI-RADS (regex)
    # ========================================================================
    try:
        r_birads = extraer_birads(full_report)
    except Exception as e:
        return _resultado_de_error(informe_id, "extraer_birads", e)

    # ========================================================================
    # PASO 2: Verificar BI-RADS con ML (inmediato tras el regex)
    # ========================================================================
    if usar_verificador_ml:
        try:
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
        texto_info = extraer_texto_recomendacion(
            recommendations_col=recommendations_col,
            full_report=full_report,
        )

        if texto_info["encontrado"]:
            r_rec = clasificar_recomendacion(
                texto_info["texto_normalizado"],
                es_ya_normalizado=True,
            )
            # Asegurar que la trazabilidad incluya el texto original
            r_rec["trazabilidad"]["texto_original"] = texto_info["texto"]
        else:
            # Sin bloque de recomendaciones → r_rec vacío (cotejo dirá no_procesable)
            r_rec = {
                "categorias_detectadas": [],
                "categoria_principal": None,
                "confianza": "no_clasificada",
                "metodo": None,
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
        "--no-ml",
        action="store_true",
        help="Desactiva el verificador ML (más rápido).",
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
        usar_verificador_ml=not args.no_ml,
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
            "nombre": "T3: Alerta alta (BI-RADS 4 + criterio medico)",
            "input": {"full_report": INFORME_ALERTA_ALTA, "informe_id": "test_T3"},
            "esperado": {
                "birads.valor": 4,
                "cotejo_acr.estado": "incoherente",
                "cotejo_acr.alerta": True,
                "cotejo_acr.severidad": "alta",
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
                # Se espera que no lance excepción y el cotejo diga no_procesable o similar
                # No validamos campos específicos, solo que devuelva un dict válido
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
