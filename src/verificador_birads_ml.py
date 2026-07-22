"""
Módulo 4: Apoyo de lectura del BI-RADS con DistilBETO

Actúa como APOYO A LA LECTURA de la categoría BI-RADS declarada, no como un
juez clínico. El modelo DistilBETO (notebook 04) lee la mención BI-RADS en su
contexto local y aporta una lectura de respaldo que refuerza (o cuestiona) la
extracción por reglas.

Fundamento empírico (ablación por enmascaramiento, notebook 04b / ablacion):
- Con el número visible en el texto, el modelo lo lee bien (Macro F1 ≈0.94).
- Sin el número, no reconstruye la categoría desde los hallazgos (≈0.54).
- Conclusión: el modelo es un LECTOR del BI-RADS declarado, NO un evaluador
  clínico independiente. Por eso su rol es de apoyo a la lectura.

Filosofía (rol de apoyo a la lectura):
- La extracción literal (regex + buscador híbrido) es la AUTORIDAD: es
  determinista y auditable.
- El ML es un APOYO DE LECTURA: cuando concuerda, refuerza la confianza en la
  extracción; cuando difiere, señala una lectura incierta a revisar. NUNCA
  sobrescribe el texto literal ni genera por sí solo una alerta clínica.
- En inferencia, el ML lee una VENTANA LOCAL alrededor de la mención candidata
  (no clasifica el informe completo): es una relectura en contexto, no un
  diagnóstico.

Uso como librería:

    from src.verificador_birads_ml import verificar_extraccion_birads
    resultado = verificar_extraccion_birads(
        full_report="...",
        birads_regex=4,
        confianza_regex="alta"
    )

Versión mejorada con búsqueda híbrida (v2):

    from src.verificador_birads_ml import verificar_extraccion_birads_con_buscador
    resultado = verificar_extraccion_birads_con_buscador(
        full_report="...",
        birads_regex=4,
        confianza_regex="alta"
    )

Autor: Sebastián Inostroza (BME513, U. de Valparaíso)
Fecha: 20 de junio de 2026
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, Optional

import numpy as np


# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

# Ruta al modelo entrenado en el notebook 04
_RUTA_MODELO_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "results_distilbeto",
    "best_model"
)

# Umbral de confianza mínima para que el ML pueda "opinar"
UMBRAL_CONFIANZA_ML = 0.60

# Umbral para considerar que el ML tiene ALTA confianza
UMBRAL_CONFIANZA_ML_ALTA = 0.75

# Estados posibles del apoyo de lectura.
# Nota: describen la CONCORDANCIA entre la lectura literal (regex) y la lectura
# de apoyo (ML). No expresan un juicio clínico sobre la categoría.
ESTADOS_VERIFICACION = [
    "confirmado",           # Regex alta + lectura ML concuerda (refuerza confianza)
    "confirmado_doble",     # Regex media/baja + lectura ML concuerda (recupera confianza)
    "ml_no_confirma",       # Regex alta + lectura ML difiere → prima la lectura literal
    "discrepante_real",     # Regex media/baja + lectura ML difiere → lectura incierta, revisar
    "ml_inseguro",          # Lectura ML sin confianza suficiente para apoyar
    "no_verificable",       # No hay mención legible para apoyar la lectura
    "alerta_omision_buscador",  # buscador detectó omisión de BI-RADS
]


# =============================================================================
# LAZY LOADING DEL MODELO
# =============================================================================

_MODELO_CARGADO = {
    "tokenizer": None,
    "modelo": None,
    "device": None,
}


def _cargar_modelo(ruta_modelo: Optional[str] = None) -> None:
    """Carga el modelo DistilBETO fine-tuneado (lazy loading)."""
    if _MODELO_CARGADO["modelo"] is not None:
        return

    ruta = ruta_modelo or _RUTA_MODELO_DEFAULT

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el modelo en: {ruta}\n"
            f"Ejecuta primero el notebook 04 para entrenar el modelo."
        )

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise ImportError(
            "Se requieren torch y transformers instalados. "
            f"Error original: {e}"
        )

    print(f"Cargando modelo desde: {ruta}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta)

    # Determinar device (MPS para Apple Silicon, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    modelo = modelo.to(device)
    modelo.eval()

    _MODELO_CARGADO["tokenizer"] = tokenizer
    _MODELO_CARGADO["modelo"] = modelo
    _MODELO_CARGADO["device"] = device

    print(f"Modelo cargado en device: {device}", file=sys.stderr)


# =============================================================================
# EXTRACCIÓN DEL BLOQUE CONCLUSIÓN
# =============================================================================

def _extraer_bloque_conclusion(full_report: str) -> Optional[str]:
    """
    Extrae el bloque de conclusion del informe usando regex.

    Reutiliza la lógica del extractor_birads.py (patrones de encabezado
    de conclusion).

    Args:
        full_report: Texto completo del informe

    Returns:
        Texto del bloque conclusion o None si no se detecta
    """
    # Patrones de encabezado de conclusion (mismos del extractor_birads.py)
    PATRONES_ENCABEZADO_CONCLUSION = [
        r"CONCLUSI[ÓO]N\s*[:.]?",
        r"CONCLUSIONES\s*[:.]?",
        r"IMPRESI[ÓO]N\s+DIAGN[ÓO]STICA\s*[:.]?",
        r"IMPRESION\s+DIAGNOSTICA\s*[:.]?",
        r"DIAGN[ÓO]STICO\s+FINAL\s*[:.]?",
        r"HALLAZGOS?\s+Y\s+CONCLUSIONES?\s*[:.]?",
    ]

    for patron in PATRONES_ENCABEZADO_CONCLUSION:
        match = re.search(patron, full_report, re.IGNORECASE)
        if match:
            # Extraer desde después del encabezado hasta el final del informe
            inicio = match.end()
            texto_conclusion = full_report[inicio:].strip()

            # Limitar a las primeras 500 palabras del bloque
            palabras = texto_conclusion.split()
            if len(palabras) > 500:
                texto_conclusion = " ".join(palabras[:500])

            return texto_conclusion

    return None


# =============================================================================
# PREDICCIÓN CON EL MODELO
# =============================================================================

def _predecir_birads_ml(texto: str) -> Dict[str, Any]:
    """
    Predice el BI-RADS usando DistilBETO sobre un texto dado.

    Args:
        texto: Texto sobre el cual predecir (típicamente el bloque conclusion)

    Returns:
        Dict con:
            - birads_predicho: int (0-6)
            - confianza_maxima: float (0-1)
            - distribucion_probabilidades: Dict[int, float]
    """
    import torch
    import torch.nn.functional as F

    if _MODELO_CARGADO["modelo"] is None:
        _cargar_modelo()

    tokenizer = _MODELO_CARGADO["tokenizer"]
    modelo = _MODELO_CARGADO["modelo"]
    device = _MODELO_CARGADO["device"]

    # Tokenizar
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predecir
    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
        probabilidades = F.softmax(logits, dim=-1).cpu().numpy()[0]

    birads_predicho = int(np.argmax(probabilidades))
    confianza_maxima = float(probabilidades[birads_predicho])

    distribucion = {i: float(prob) for i, prob in enumerate(probabilidades)}

    return {
        "birads_predicho": birads_predicho,
        "confianza_maxima": confianza_maxima,
        "distribucion_probabilidades": distribucion,
    }


# =============================================================================
# LÓGICA DE DETERMINACIÓN DE ESTADOS (v2.1 - JERÁRQUICA)
# =============================================================================

def _determinar_estado_verificacion(
    birads_regex: Optional[int],
    confianza_regex: str,
    birads_ml: int,
    confianza_ml: float,
) -> Dict[str, Any]:
    """
    Determina la CONCORDANCIA entre la lectura literal (regex) y la lectura de
    apoyo (ML), usando lógica jerárquica.

    Encuadre (apoyo de lectura, NO juicio clínico):
    - La lectura literal (regex + buscador) es la AUTORIDAD: determinista y auditable.
    - El ML es un APOYO DE LECTURA: refuerza cuando concuerda, señala lectura
      incierta cuando difiere. Nunca sobrescribe el texto ni emite alerta clínica.

    Args:
        birads_regex: Valor extraído por regex (0-6) o None si no se extrajo
        confianza_regex: "alta" | "media" | "baja" | "no_detectado"
        birads_ml: Valor leído por el ML (0-6)
        confianza_ml: Confianza de la lectura ML (0-1)

    Returns:
        Dict con: estado, regla, mensaje
    """
    coincide = (birads_regex == birads_ml)

    # ==================== REGLA 1: lectura ML sin confianza ====================
    if confianza_ml < UMBRAL_CONFIANZA_ML:
        return {
            "estado": "ml_inseguro",
            "regla": "regla_1_ml_baja_confianza",
            "mensaje": (
                f"La lectura de apoyo (ML) tiene baja confianza "
                f"({confianza_ml:.2f} < {UMBRAL_CONFIANZA_ML}); no aporta respaldo "
                f"a la lectura literal en este caso."
            )
        }

    # ==================== REGLA 2: Regex ALTA + lectura ML concuerda ====================
    if confianza_regex == "alta" and coincide:
        return {
            "estado": "confirmado",
            "regla": "regla_2_regex_alta_ml_coincide",
            "mensaje": (
                f"Lectura literal (regex, confianza alta): BI-RADS {birads_regex}. "
                f"La lectura de apoyo (ML) concuerda ({confianza_ml:.2f}), "
                f"reforzando la confianza en la extracción."
            )
        }

    # ==================== REGLA 3: Regex ALTA + lectura ML difiere ====================
    if confianza_regex == "alta" and not coincide:
        return {
            "estado": "ml_no_confirma",
            "regla": "regla_3_regex_alta_ml_disiente",
            "mensaje": (
                f"Lectura literal (regex, confianza alta): BI-RADS {birads_regex} "
                f"— es la autoridad y se mantiene. La lectura de apoyo (ML) leyó "
                f"BI-RADS {birads_ml} ({confianza_ml:.2f}); al ser un apoyo, no "
                f"altera la extracción literal. Sin implicancia clínica."
            )
        }

    # ==================== REGLA 4: Regex MEDIA/BAJA + lectura ML concuerda ====================
    if confianza_regex in ("media", "baja") and coincide:
        return {
            "estado": "confirmado_doble",
            "regla": "regla_4_regex_media_ml_confirma",
            "mensaje": (
                f"Lectura literal (regex, confianza {confianza_regex}): BI-RADS {birads_regex}. "
                f"La lectura de apoyo (ML) coincide ({confianza_ml:.2f}) y RECUPERA "
                f"confianza sobre una extracción que por sí sola era incierta."
            )
        }

    # ==================== REGLA 5: Regex MEDIA/BAJA + lectura ML difiere ====================
    if confianza_regex in ("media", "baja") and not coincide:
        if confianza_ml >= UMBRAL_CONFIANZA_ML_ALTA:
            return {
                "estado": "discrepante_real",
                "regla": "regla_5a_discrepancia_ml_alta",
                "mensaje": (
                    f"Lectura incierta del BI-RADS: la extracción literal "
                    f"(confianza {confianza_regex}) leyó BI-RADS {birads_regex}, y la "
                    f"lectura de apoyo (ML, {confianza_ml:.2f}) leyó BI-RADS {birads_ml}. "
                    f"Ambas lecturas difieren y ninguna es concluyente: conviene "
                    f"verificar manualmente qué categoría declara el informe."
                )
            }
        else:
            return {
                "estado": "ml_no_confirma",
                "regla": "regla_5b_regex_media_ml_media",
                "mensaje": (
                    f"Lectura literal (regex, confianza {confianza_regex}): BI-RADS {birads_regex}. "
                    f"La lectura de apoyo (ML, {confianza_ml:.2f}) difiere pero sin "
                    f"confianza suficiente para respaldar otra lectura; se mantiene la "
                    f"extracción literal. Sin implicancia clínica."
                )
            }

    # ==================== CASO EDGE: sin BI-RADS por regex ====================
    if birads_regex is None:
        return {
            "estado": "no_verificable",
            "regla": "regla_edge_sin_regex",
            "mensaje": (
                f"La extracción literal no encontró BI-RADS. La lectura de apoyo "
                f"(ML) leyó BI-RADS {birads_ml} ({confianza_ml:.2f}); es solo una "
                f"lectura de respaldo, a confirmar por el revisor."
            )
        }

    # ==================== FALLBACK (no debería ocurrir) ====================
    return {
        "estado": "no_verificable",
        "regla": "fallback",
        "mensaje": "Caso no contemplado en la lógica de apoyo de lectura."
    }


# =============================================================================
# FUNCIÓN PÚBLICA PRINCIPAL (versión original - se mantiene retrocompatible)
# =============================================================================

def verificar_extraccion_birads(
    full_report: str,
    birads_regex: Optional[int],
    confianza_regex: str,
    ruta_modelo: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verifica la extracción del extractor regex usando DistilBETO como
    segunda opinión.

    Filosofía v2.1: Regex es autoridad clínica, ML es sanity check.

    Args:
        full_report: Texto completo del informe mamográfico
        birads_regex: Valor BI-RADS extraído por regex (0-6) o None
        confianza_regex: "alta" | "media" | "baja" | "no_detectado"
        ruta_modelo: Ruta opcional al modelo (default: notebooks/results_distilbeto/best_model)

    Returns:
        Dict con:
            - birads_regex: int | None (input original)
            - confianza_regex: str (input original)
            - birads_predicho_ml: int | None
            - confianza_ml: float
            - distribucion_probabilidades: Dict[int, float]
            - coincide_con_regex: bool
            - estado_verificacion: str (ver ESTADOS_VERIFICACION)
            - regla_aplicada: str (para trazabilidad)
            - mensaje: str (explicación humana)
    """
    # Cargar modelo si no está cargado
    if _MODELO_CARGADO["modelo"] is None:
        _cargar_modelo(ruta_modelo)

    # Extraer bloque conclusion
    bloque_conclusion = _extraer_bloque_conclusion(full_report)

    if bloque_conclusion is None:
        return {
            "birads_regex": birads_regex,
            "confianza_regex": confianza_regex,
            "birads_predicho_ml": None,
            "confianza_ml": 0.0,
            "distribucion_probabilidades": {},
            "coincide_con_regex": False,
            "estado_verificacion": "no_verificable",
            "regla_aplicada": "no_hay_bloque_conclusion",
            "mensaje": (
                "No se detectó un bloque de conclusion en el informe. "
                "No se puede aplicar el verificador ML."
            )
        }

    # Predecir con ML sobre el bloque conclusion
    prediccion_ml = _predecir_birads_ml(bloque_conclusion)

    birads_ml = prediccion_ml["birads_predicho"]
    confianza_ml = prediccion_ml["confianza_maxima"]

    # Determinar estado de verificación
    estado_info = _determinar_estado_verificacion(
        birads_regex=birads_regex,
        confianza_regex=confianza_regex,
        birads_ml=birads_ml,
        confianza_ml=confianza_ml,
    )

    return {
        "birads_regex": birads_regex,
        "confianza_regex": confianza_regex,
        "birads_predicho_ml": birads_ml,
        "confianza_ml": confianza_ml,
        "distribucion_probabilidades": prediccion_ml["distribucion_probabilidades"],
        "coincide_con_regex": (birads_regex == birads_ml),
        "estado_verificacion": estado_info["estado"],
        "regla_aplicada": estado_info["regla"],
        "mensaje": estado_info["mensaje"],
        "modo": "bloque_conclusion",  # NUEVO en v2: identifica método usado
    }


# =============================================================================
# FUNCIÓN PÚBLICA v2 - USA BUSCADOR HÍBRIDO (NUEVA)
# =============================================================================

def verificar_extraccion_birads_con_buscador(
    full_report: str,
    birads_regex: Optional[int],
    confianza_regex: str,
    ruta_modelo: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verificación mejorada: usa el buscador híbrido para localizar el BI-RADS
    final del informe, en vez de depender de un bloque CONCLUSIÓN explícito.

    Ventajas sobre verificar_extraccion_birads:
    - Maneja informes SIN encabezado de conclusión
    - Detecta menciones históricas y las descarta
    - Emite alerta de omisión si hay hallazgos pero no hay BI-RADS asignado
    - Aplica DistilBETO sobre el contexto real de la conclusión

    Args:
        full_report: Texto completo del informe mamográfico
        birads_regex: Valor BI-RADS extraído por regex (0-6) o None
        confianza_regex: "alta" | "media" | "baja" | "no_detectado"
        ruta_modelo: Ruta opcional al modelo

    Returns:
        Dict compatible con verificar_extraccion_birads + campos nuevos:
            - modo: "busqueda_hibrida"
            - posicion_relativa_mencion: float (0-1)
            - alerta_omision: Dict (solo si aplica)
    """
    try:
        from src.buscador_birads import buscar_birads_final
    except ImportError as e:
        # Fallback si el buscador no está disponible
        print(
            f"[verificador_ml] Advertencia: buscador_birads no disponible ({e}). "
            f"Usando método antiguo (bloque conclusión).",
            file=sys.stderr,
        )
        return verificar_extraccion_birads(
            full_report, birads_regex, confianza_regex, ruta_modelo
        )

    # =====================================================================
    # PASO 1: Ejecutar buscador híbrido (sin ML para determinismo)
    # =====================================================================
    resultado_buscador = buscar_birads_final(
        full_report,
        usar_ml_si_ambiguo=False,
    )

    # =====================================================================
    # PASO 2: Si el buscador emite alerta de omisión, propagarla
    # =====================================================================
    if "alerta" in resultado_buscador:
        alerta = resultado_buscador["alerta"]
        return {
            "birads_regex": birads_regex,
            "confianza_regex": confianza_regex,
            "birads_predicho_ml": None,
            "confianza_ml": 0.0,
            "distribucion_probabilidades": {},
            "coincide_con_regex": False,
            "estado_verificacion": "alerta_omision_buscador",
            "regla_aplicada": "buscador_hibrido_omision",
            "mensaje": alerta["mensaje"],
            "modo": "busqueda_hibrida",
            "alerta_omision": alerta,
            "razon_buscador": resultado_buscador.get("razon_no_detectado"),
        }

    # =====================================================================
    # PASO 3: Si el buscador no encontró nada (sin alerta), fallback antiguo
    # =====================================================================
    mencion = resultado_buscador.get("mencion_seleccionada")
    if mencion is None:
        # No hay mención válida y no hay alerta → informe muy corto o irrelevante
        return {
            "birads_regex": birads_regex,
            "confianza_regex": confianza_regex,
            "birads_predicho_ml": None,
            "confianza_ml": 0.0,
            "distribucion_probabilidades": {},
            "coincide_con_regex": False,
            "estado_verificacion": "no_verificable",
            "regla_aplicada": "buscador_hibrido_sin_menciones",
            "mensaje": (
                "El buscador híbrido no encontró menciones BI-RADS válidas "
                "en el informe."
            ),
            "modo": "busqueda_hibrida",
        }

    # =====================================================================
    # PASO 4: Cargar modelo si no está cargado
    # =====================================================================
    if _MODELO_CARGADO["modelo"] is None:
        _cargar_modelo(ruta_modelo)

    # =====================================================================
    # PASO 5: Extraer contexto ampliado alrededor de la mención
    #         El ML analiza este fragmento en vez del bloque CONCLUSIÓN
    # =====================================================================
    posicion_inicio = mencion["posicion_inicio"]
    posicion_fin = mencion["posicion_fin"]

    inicio_contexto = max(0, posicion_inicio - 200)
    fin_contexto = min(len(full_report), posicion_fin + 50)
    fragmento_contexto = full_report[inicio_contexto:fin_contexto].strip()

    # Fallback si el fragmento es demasiado corto
    if len(fragmento_contexto) < 30:
        fragmento_contexto = full_report[max(0, posicion_inicio - 500):]

    # =====================================================================
    # PASO 6: Predecir con ML sobre el fragmento
    # =====================================================================
    prediccion_ml = _predecir_birads_ml(fragmento_contexto)

    birads_ml = prediccion_ml["birads_predicho"]
    confianza_ml = prediccion_ml["confianza_maxima"]

    # =====================================================================
    # PASO 7: Determinar estado usando la lógica existente
    # =====================================================================
    estado_info = _determinar_estado_verificacion(
        birads_regex=birads_regex,
        confianza_regex=confianza_regex,
        birads_ml=birads_ml,
        confianza_ml=confianza_ml,
    )

    return {
        "birads_regex": birads_regex,
        "confianza_regex": confianza_regex,
        "birads_predicho_ml": birads_ml,
        "confianza_ml": confianza_ml,
        "distribucion_probabilidades": prediccion_ml["distribucion_probabilidades"],
        "coincide_con_regex": (birads_regex == birads_ml),
        "estado_verificacion": estado_info["estado"],
        "regla_aplicada": estado_info["regla"],
        "mensaje": estado_info["mensaje"],
        "modo": "busqueda_hibrida",
        "posicion_relativa_mencion": mencion["posicion_relativa"],
        "birads_localizado_por_buscador": mencion["valor"],
    }


# =============================================================================
# TESTS INLINE
# =============================================================================

def _ejecutar_tests() -> None:
    """Suite de tests inline. Ejecutar con: python -m src.verificador_birads_ml"""
    print("=" * 60)
    print("TESTS: verificador_birads_ml.py")
    print("=" * 60)

    # ============= Tests de _determinar_estado_verificacion =============
    print("\n1. Tests de lógica de estados (v2.1)...")

    casos = [
        {
            "nombre": "Regex ALTA + ML coincide",
            "birads_regex": 4, "confianza_regex": "alta",
            "birads_ml": 4, "confianza_ml": 0.85,
            "estado_esperado": "confirmado"
        },
        {
            "nombre": "Regex ALTA + ML discrepa (regex gana)",
            "birads_regex": 4, "confianza_regex": "alta",
            "birads_ml": 2, "confianza_ml": 0.65,
            "estado_esperado": "ml_no_confirma"
        },
        {
            "nombre": "Regex MEDIA + ML confirma",
            "birads_regex": 2, "confianza_regex": "media",
            "birads_ml": 2, "confianza_ml": 0.72,
            "estado_esperado": "confirmado_doble"
        },
        {
            "nombre": "Regex BAJA + ML discrepa ALTA confianza (discrepancia real)",
            "birads_regex": 3, "confianza_regex": "baja",
            "birads_ml": 4, "confianza_ml": 0.88,
            "estado_esperado": "discrepante_real"
        },
        {
            "nombre": "Regex BAJA + ML discrepa MEDIA confianza (regex gana)",
            "birads_regex": 3, "confianza_regex": "baja",
            "birads_ml": 4, "confianza_ml": 0.68,
            "estado_esperado": "ml_no_confirma"
        },
        {
            "nombre": "ML sin confianza",
            "birads_regex": 4, "confianza_regex": "alta",
            "birads_ml": 3, "confianza_ml": 0.45,
            "estado_esperado": "ml_inseguro"
        },
    ]

    pasados_estados = 0
    for caso in casos:
        resultado = _determinar_estado_verificacion(
            birads_regex=caso["birads_regex"],
            confianza_regex=caso["confianza_regex"],
            birads_ml=caso["birads_ml"],
            confianza_ml=caso["confianza_ml"],
        )
        ok = resultado["estado"] == caso["estado_esperado"]
        marca = "✓" if ok else "✗"
        if ok:
            pasados_estados += 1
        print(f"  [{marca}] {caso['nombre']}: {resultado['estado']}")
        if not ok:
            print(f"       Esperado: {caso['estado_esperado']}")

    print(f"\n  Resultado: {pasados_estados}/{len(casos)} tests de estados pasados")

    # ============= Tests de _extraer_bloque_conclusion =============
    print("\n2. Tests de extracción de bloque conclusion...")

    tests_extraccion = [
        {
            "nombre": "Con encabezado CONCLUSIÓN",
            "informe": "HALLAZGOS:\nnormales.\n\nCONCLUSIÓN:\nBI-RADS 2.",
            "debe_extraer": True,
        },
        {
            "nombre": "Con encabezado IMPRESIÓN DIAGNÓSTICA",
            "informe": "Estudio X.\n\nIMPRESIÓN DIAGNÓSTICA:\nCategoría 3.",
            "debe_extraer": True,
        },
        {
            "nombre": "Sin encabezado (informe minimalista)",
            "informe": "Mamografía normal. BI-RADS 1.",
            "debe_extraer": False,
        },
    ]

    pasados_extraccion = 0
    for test in tests_extraccion:
        resultado = _extraer_bloque_conclusion(test["informe"])
        extraido = resultado is not None
        ok = extraido == test["debe_extraer"]
        marca = "✓" if ok else "✗"
        if ok:
            pasados_extraccion += 1
        print(f"  [{marca}] {test['nombre']}")
        if resultado:
            print(f"       Extraído: '{resultado[:60]}...'")

    print(f"\n  Resultado: {pasados_extraccion}/{len(tests_extraccion)} tests de extracción pasados")

    # ============= Resumen =============
    total_pasados = pasados_estados + pasados_extraccion
    total = len(casos) + len(tests_extraccion)
    print("\n" + "=" * 60)
    print(f"RESUMEN: {total_pasados}/{total} tests pasados")
    print("=" * 60)

    if total_pasados == total:
        print("✓ Todos los tests pasaron")
    else:
        print("✗ Algunos tests fallaron")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    _ejecutar_tests()
