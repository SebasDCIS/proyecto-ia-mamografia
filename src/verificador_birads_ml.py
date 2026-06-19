"""
Verificador secundario de extracción BI-RADS usando DistilBETO.

Módulo del MVP del proyecto BME513 (Universidad de Valparaíso).

OBJETIVO
========
Verificar la calidad técnica de la extracción regex del BI-RADS realizada
por `extractor_birads.py` mediante una segunda señal independiente generada
por el modelo DistilBETO fine-tuneado en el notebook 04.

FILOSOFÍA
=========
El extractor regex captura LITERALMENTE lo que el radiólogo escribió en
la conclusión. Es la fuente primaria de verdad. El verificador ML solo
tiene voz importante cuando la regex no está segura.

Jerarquía de fuentes:
    1. Regex = lo que el radiólogo escribió (autoridad clínica)
    2. ML    = patrón estadístico aprendido (segunda opinión técnica)

LO QUE ESTE MÓDULO NO HACE
==========================
- NO genera alertas clínicas (eso lo hace cotejo_acr.py)
- NO detecta inconsistencias entre hallazgos y conclusión (eso lo cubre
  el proyecto complementario coherence-audit)
- NO reemplaza la regex como fuente primaria

LÓGICA DE VERIFICACIÓN (v2.1)
=============================
Cinco estados según la confianza de cada fuente:

| Estado            | Cuándo aplica                          | Acción              |
|-------------------|----------------------------------------|---------------------|
| confirmado        | regex alta + ML coincide               | Sin acción          |
| confirmado_doble  | regex media/baja + ML confirma         | Sin acción          |
| ml_no_confirma    | regex alta + ML discrepa               | Se prioriza regex   |
| discrepante_real  | regex media/baja + ML discrepa         | REVISIÓN MANUAL     |
| ml_inseguro       | regex media/baja + ML sin confianza    | No verificable      |

DECISIÓN METODOLÓGICA
=====================
DistilBETO se aplica SOLO sobre el bloque CONCLUSIÓN, no sobre el
Full_Report completo. Esto garantiza:

1. Especificidad: el modelo y la regex miran el mismo texto, así que
   cualquier discrepancia es un problema de extracción, no de razonamiento
   clínico (eso lo cubre coherence-audit).

2. Honestidad metodológica: el modelo fue entrenado con el informe completo
   y aplicarlo sobre solo la conclusión reduce su F1 esperado (de 0.9386
   en entrenamiento a 0.8987 sobre solo CONCLUSIÓN). Este costo se asume
   conscientemente.

MODELO
======
- Base: dccuchile/distilbert-base-spanish-uncased (DistilBETO)
- Fine-tuning: 3 épocas + augmentación textual (notebook 04)
- Macro F1 (validación): 0.9386
- Checkpoint: results_distilbeto/checkpoint-1743
  (identificado por HuggingFace Trainer como best_model_checkpoint)

USO
===
    from src.verificador_birads_ml import verificar_extraccion_birads

    resultado = verificar_extraccion_birads(
        full_report=texto_informe,
        birads_regex=2,
        confianza_regex="alta",
    )

    print(resultado["estado_verificacion"])  # ej: 'confirmado'
    print(resultado["mensaje"])              # mensaje descriptivo

Autor: Sebastián Inostroza Hurtado
Curso: BME513 - Inteligencia Artificial en Salud (DCIS, UV)
Fecha: Junio 2026
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

_RUTA_MODELO_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "results_distilbeto",
    "checkpoint-1743",
)

_MAX_LENGTH = 256
_UMBRAL_CONFIANZA_ML_ALTA = 0.70
_UMBRAL_CONFIANZA_ML_BAJA = 0.50


# =============================================================================
# CARGA PEREZOSA DEL MODELO
# =============================================================================

_modelo = None
_tokenizer = None
_device = None


def _cargar_modelo(ruta_modelo: Optional[str] = None) -> None:
    """Carga el modelo y tokenizer. Solo se ejecuta una vez."""
    global _modelo, _tokenizer, _device

    if _modelo is not None:
        return

    ruta = ruta_modelo or _RUTA_MODELO_DEFAULT

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el modelo en: {ruta}\n"
            f"Verifica que el directorio contenga config.json + "
            f"model.safetensors + tokenizer."
        )

    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _tokenizer = AutoTokenizer.from_pretrained(ruta)
    _modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
    _modelo.to(_device)
    _modelo.eval()


# =============================================================================
# FUNCIONES INTERNAS
# =============================================================================

_PATRON_INICIO_CONCLUSION = re.compile(
    r"\b(conclusi[oó]n|valoraci[oó]n)\s*:?",
    re.IGNORECASE,
)
_PATRON_FIN_BLOQUE = re.compile(
    r"\b(recomendaciones?|sugerencias?|indicaciones?|firma|atte|atentamente|cordialmente|dr\.|dra\.)\s*:?",
    re.IGNORECASE,
)


def _extraer_bloque_conclusion(full_report: str) -> Optional[str]:
    """Extrae el texto del bloque CONCLUSIÓN del informe."""
    if not isinstance(full_report, str) or not full_report.strip():
        return None

    match_inicio = _PATRON_INICIO_CONCLUSION.search(full_report)
    if not match_inicio:
        return None

    inicio = match_inicio.end()
    resto = full_report[inicio:]
    match_fin = _PATRON_FIN_BLOQUE.search(resto)
    fin = inicio + match_fin.start() if match_fin else len(full_report)

    bloque = full_report[inicio:fin].strip()
    return bloque if bloque else None


def _predecir_birads_ml(texto: str) -> Dict[str, Any]:
    """Predice BI-RADS desde un texto usando DistilBETO."""
    _cargar_modelo()

    if not isinstance(texto, str) or not texto.strip():
        return {"birads_predicho": None, "confianza": 0.0, "distribucion": {}}

    inputs = _tokenizer(
        texto,
        truncation=True,
        padding=True,
        max_length=_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _modelo(**inputs)
        probs = torch.nn.functional.softmax(
            outputs.logits, dim=-1
        ).squeeze().cpu().numpy()

    birads_predicho = int(probs.argmax())
    confianza = float(probs[birads_predicho])
    distribucion = {i: round(float(probs[i]), 4) for i in range(len(probs))}

    return {
        "birads_predicho": birads_predicho,
        "confianza": round(confianza, 4),
        "distribucion": distribucion,
    }


def _determinar_estado_verificacion(
    birads_regex: int,
    confianza_regex: str,
    birads_ml: int,
    confianza_ml: float,
) -> Dict[str, Any]:
    """Aplica la lógica v2.1 para determinar el estado de la verificación.

    Args:
        birads_regex: BI-RADS extraído por la regex.
        confianza_regex: 'alta' | 'media' | 'baja'.
        birads_ml: BI-RADS predicho por DistilBETO.
        confianza_ml: probabilidad de la clase predicha (0.0 a 1.0).
    """
    coinciden = birads_regex == birads_ml
    ml_alta = confianza_ml >= _UMBRAL_CONFIANZA_ML_ALTA
    ml_baja = confianza_ml < _UMBRAL_CONFIANZA_ML_BAJA
    regex_confiable = confianza_regex == "alta"

    # CASOS DE REGEX ALTA CONFIANZA
    if regex_confiable:
        if coinciden:
            if ml_alta:
                return {
                    "estado": "confirmado",
                    "mensaje": (
                        f"Regex (confianza alta) y ML (confianza alta "
                        f"{confianza_ml:.2f}) coinciden en BI-RADS "
                        f"{birads_regex}. Extracción validada."
                    ),
                    "regla": "regla_1_doble_confirmacion_alta",
                }
            else:
                return {
                    "estado": "confirmado",
                    "mensaje": (
                        f"Regex (confianza alta) extrajo BI-RADS "
                        f"{birads_regex}. ML coincide con confianza "
                        f"moderada ({confianza_ml:.2f}). Se confía en "
                        f"la extracción regex."
                    ),
                    "regla": "regla_2_regex_alta_ml_modera",
                }
        else:
            return {
                "estado": "ml_no_confirma",
                "mensaje": (
                    f"Regex (confianza alta) extrajo BI-RADS "
                    f"{birads_regex}. ML predijo BI-RADS {birads_ml} "
                    f"con confianza {confianza_ml:.2f}. Se prioriza "
                    f"la extracción regex (literal del informe). "
                    f"El ML puede haber confundido un patrón "
                    f"estadístico. Sin alerta clínica."
                ),
                "regla": "regla_3_regex_alta_ml_disiente",
            }

    # CASOS DE REGEX MEDIA/BAJA CONFIANZA
    if ml_baja:
        return {
            "estado": "ml_inseguro",
            "mensaje": (
                f"Regex (confianza {confianza_regex}) extrajo BI-RADS "
                f"{birads_regex}. ML no tuvo confianza suficiente "
                f"({confianza_ml:.2f}) para verificar. No verificable "
                f"por verificador dual."
            ),
            "regla": "regla_4_ambos_inseguros",
        }

    if coinciden:
        return {
            "estado": "confirmado_doble",
            "mensaje": (
                f"Regex (confianza {confianza_regex}) extrajo BI-RADS "
                f"{birads_regex}. ML lo confirma con confianza "
                f"{confianza_ml:.2f}. Validación cruzada exitosa."
            ),
            "regla": "regla_5_validacion_cruzada",
        }

    return {
        "estado": "discrepante_real",
        "mensaje": (
            f"Texto del informe presenta formato atípico o ambigüedad. "
            f"Regex (confianza {confianza_regex}) extrajo BI-RADS "
            f"{birads_regex}. ML predijo BI-RADS {birads_ml} con alta "
            f"confianza ({confianza_ml:.2f}). Se recomienda revisión "
            f"manual para confirmar la extracción."
        ),
        "regla": "regla_6_discrepancia_real",
    }


# =============================================================================
# FUNCIÓN PÚBLICA PRINCIPAL
# =============================================================================

def verificar_extraccion_birads(
    full_report: str,
    birads_regex: Optional[int],
    confianza_regex: str,
) -> Dict[str, Any]:
    """Verifica la extracción regex del BI-RADS usando DistilBETO.

    Aplica el modelo DistilBETO sobre el bloque CONCLUSIÓN del informe
    para verificar si la regex extrajo correctamente el BI-RADS.
    Implementa la lógica v2.1 que prioriza la regex cuando es de alta
    confianza, reflejando que el radiólogo (no el modelo) es la autoridad
    clínica.

    NO genera alertas clínicas. Solo etiquetas de calidad de extracción.

    Args:
        full_report: texto completo del informe.
        birads_regex: BI-RADS extraído por extractor_birads (0-6 o None).
        confianza_regex: 'alta' | 'media' | 'baja'.

    Returns:
        Dict con:
            - birads_predicho_ml: BI-RADS predicho por DistilBETO
            - confianza_ml: probabilidad de la clase predicha
            - distribucion_completa: probabilidades por cada clase
            - bloque_conclusion: texto del bloque (truncado a 200 chars)
            - coincide_con_regex: bool
            - estado_verificacion: 'confirmado' | 'confirmado_doble' |
                'ml_no_confirma' | 'discrepante_real' | 'ml_inseguro' |
                'no_verificable'
            - mensaje: interpretación descriptiva del estado
            - regla_aplicada: cuál regla resolvió el caso
            - afecta_confiabilidad_tecnica: bool, si debe alterar el
              nivel de confiabilidad reportado por el cotejo
    """
    if birads_regex is None:
        return {
            "birads_predicho_ml": None,
            "confianza_ml": 0.0,
            "distribucion_completa": {},
            "bloque_conclusion": None,
            "coincide_con_regex": None,
            "estado_verificacion": "no_verificable",
            "mensaje": (
                "La regex no extrajo un BI-RADS de la conclusión, "
                "por lo que no hay un valor a verificar contra el modelo ML."
            ),
            "regla_aplicada": None,
            "afecta_confiabilidad_tecnica": False,
        }

    bloque = _extraer_bloque_conclusion(full_report)
    if not bloque:
        return {
            "birads_predicho_ml": None,
            "confianza_ml": 0.0,
            "distribucion_completa": {},
            "bloque_conclusion": None,
            "coincide_con_regex": None,
            "estado_verificacion": "no_verificable",
            "mensaje": (
                "No se encontró un bloque CONCLUSIÓN claro en el informe, "
                "por lo que no se puede aplicar la verificación ML."
            ),
            "regla_aplicada": None,
            "afecta_confiabilidad_tecnica": False,
        }

    prediccion = _predecir_birads_ml(bloque)
    birads_ml = prediccion["birads_predicho"]
    confianza_ml = prediccion["confianza"]

    estado = _determinar_estado_verificacion(
        birads_regex=birads_regex,
        confianza_regex=confianza_regex,
        birads_ml=birads_ml,
        confianza_ml=confianza_ml,
    )

    afecta_confiabilidad = estado["estado"] == "discrepante_real"

    return {
        "birads_predicho_ml": birads_ml,
        "confianza_ml": confianza_ml,
        "distribucion_completa": prediccion["distribucion"],
        "bloque_conclusion": bloque[:200],
        "coincide_con_regex": birads_regex == birads_ml,
        "estado_verificacion": estado["estado"],
        "mensaje": estado["mensaje"],
        "regla_aplicada": estado["regla"],
        "afecta_confiabilidad_tecnica": afecta_confiabilidad,
    }


# =============================================================================
# TESTS INLINE
# =============================================================================

def _ejecutar_tests() -> None:
    """Suite de tests inline. Ejecutar con: python -m src.verificador_birads_ml"""
    print("=" * 75)
    print("TESTS DE src/verificador_birads_ml.py (lógica v2.1)")
    print("=" * 75)

    print("\n[Setup] Cargando modelo DistilBETO...")
    try:
        _cargar_modelo()
        print(f"  ✓ Modelo cargado en device: {_device}")
    except FileNotFoundError as e:
        print(f"  ✗ ERROR: {e}")
        return

    casos = [
        {
            "nombre": "C1: BI-RADS 2 + regex alta → confirmado",
            "input": {
                "full_report": (
                    "MAMOGRAFIA BILATERAL. HALLAZGOS: Mamas con densidad mixta. "
                    "CONCLUSION: - BI-RADS 2 (segun la ACR). Hallazgos benignos. "
                    "RECOMENDACIONES: - Control mamografico anual."
                ),
                "birads_regex": 2,
                "confianza_regex": "alta",
            },
            "esperados": {
                "estado_verificacion": "confirmado",
                "afecta_confiabilidad_tecnica": False,
            },
        },
        {
            "nombre": "C2: BI-RADS 1 + regex alta → confirmado",
            "input": {
                "full_report": (
                    "MAMOGRAFIA. HALLAZGOS: Sin hallazgos relevantes. "
                    "CONCLUSION: - Estudio normal. - BI-RADS 1 (segun ACR). "
                    "RECOMENDACIONES: Control anual."
                ),
                "birads_regex": 1,
                "confianza_regex": "alta",
            },
            "esperados": {
                "estado_verificacion": "confirmado",
                "afecta_confiabilidad_tecnica": False,
            },
        },
        {
            "nombre": "C3: regex None → no_verificable",
            "input": {
                "full_report": "Texto cualquiera.",
                "birads_regex": None,
                "confianza_regex": "alta",
            },
            "esperados": {
                "estado_verificacion": "no_verificable",
                "afecta_confiabilidad_tecnica": False,
            },
        },
        {
            "nombre": "C4: Sin bloque CONCLUSION → no_verificable",
            "input": {
                "full_report": "Texto libre sin estructura.",
                "birads_regex": 3,
                "confianza_regex": "alta",
            },
            "esperados": {
                "estado_verificacion": "no_verificable",
                "afecta_confiabilidad_tecnica": False,
            },
        },
        {
            "nombre": "C5: BI-RADS 0 + regex media → válido",
            "input": {
                "full_report": (
                    "MAMOGRAFIA. HALLAZGOS: Estudio incompleto. "
                    "CONCLUSION: BI-RADS 0 (segun ACR). "
                    "RECOMENDACIONES: Vista adicional."
                ),
                "birads_regex": 0,
                "confianza_regex": "media",
            },
            "esperados": {
                "afecta_confiabilidad_tecnica": False,
            },
        },
    ]

    n_pasados = 0
    for caso in casos:
        try:
            resultado = verificar_extraccion_birads(**caso["input"])
        except Exception as e:
            print(f"\n  [ERROR] {caso['nombre']}: {e}")
            continue

        checks = []
        for clave, valor_esperado in caso["esperados"].items():
            valor_real = resultado.get(clave)
            checks.append((
                valor_real == valor_esperado,
                f"{clave}: esperado={valor_esperado}, obtenido={valor_real}",
            ))

        paso = all(ok for ok, _ in checks)
        estado_str = "PASA" if paso else "FALLA"

        if paso:
            n_pasados += 1
            print(f"\n  [{estado_str}] {caso['nombre']}")
            if resultado.get("birads_predicho_ml") is not None:
                print(
                    f"         ML: BI-RADS {resultado['birads_predicho_ml']} "
                    f"(conf {resultado['confianza_ml']:.3f}), "
                    f"estado: {resultado['estado_verificacion']}"
                )
            else:
                print(f"         estado: {resultado['estado_verificacion']}")
        else:
            print(f"\n  [{estado_str}] {caso['nombre']}")
            for ok, msg in checks:
                if not ok:
                    print(f"         {msg}")

    print(f"\n{'=' * 75}")
    print(f"Resumen: {n_pasados}/{len(casos)} tests pasados")
    if n_pasados == len(casos):
        print("Estado: OK — verificador_birads_ml listo para uso.")
    else:
        print("Estado: revisar los casos que no pasaron.")
    print("=" * 75)


if __name__ == "__main__":
    _ejecutar_tests()
