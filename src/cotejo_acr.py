"""
Cotejo BI-RADS declarado vs Recomendación según norma ACR.

Módulo del MVP del proyecto BME513 (Universidad de Valparaíso).

Recibe los resultados estructurados de los dos extractores anteriores
(extractor_birads, extractor_recomendacion) y determina si existe
coherencia clínica entre la categoría BI-RADS declarada y la
recomendación emitida, según la norma ACR.

Opcionalmente integra el resultado del verificador ML (verificador_birads_ml)
para ajustar la confiabilidad técnica del procesamiento. Cuando el ML detecta
una discrepancia real (regex de baja confianza vs ML de alta confianza), la
confiabilidad técnica del cotejo se ajusta a la baja.

Genera alertas con trazabilidad completa y reportes en tres formatos:
- Detallado: para revisión caso por caso (vista detalle del dashboard)
- Compacto: para listas/tablas (vista resumen del dashboard)
- DataFrame: para exportación masiva (CSV)

Diseño en capas (todas las funciones devuelven datos, no imprimen):

    [resultados extractores]
            ↓
    cotejar_birads_vs_recomendacion()  → dict con decisión clínica
            ↓
    [decisión: requiere alerta o no]
            ↓
    Si requiere alerta:
        - generar_reporte_alerta_detallado()  → str
        - generar_reporte_alerta_compacto()   → str
        - guardar_alerta_json()                → ruta archivo

    Para múltiples casos:
        - crear_resumen_compacto()             → dict con estadísticas
        - resumen_para_dataframe()             → pd.DataFrame para CSV

Validado sobre Vázquez Noguera et al. (2025):
    - 4 347 informes procesados (vía simulación con clasificaciones del nb06)
    - 44 alertas reales (1.0% del corpus)
    - 2 alertas críticas, 34 altas, 8 medias

Autor: Sebastián Inostroza Hurtado
Fecha: Junio 2026
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.recursos.tabla_acr import (
    JERARQUIA_URGENCIA,
    MENSAJES_INCONSISTENCIA,
    SUGERENCIAS_GENERICAS,
    TABLA_ACR,
)


# =============================================================================
# FUNCIONES INTERNAS
# =============================================================================

def _calcular_confiabilidad_tecnica(
    resultado_birads: Dict[str, Any],
    resultado_recomendacion: Dict[str, Any],
    verificacion_ml: Optional[Dict[str, Any]] = None,
) -> str:
    """Determina el nivel de confiabilidad técnica del procesamiento.

    Reglas base (sin ML):
        - 'alta':  todas las extracciones con regla estricta, sin fallback
        - 'media': alguna extracción con confianza media o usó TF-IDF
        - 'baja':  alguna extracción con confianza baja

    Ajustes por verificación ML (si verificacion_ml está presente):
        - estado 'discrepante_real': fuerza confiabilidad a 'baja'
          (el ML detectó que la regex de confianza media/baja probablemente
          extrajo mal el BI-RADS)
        - estado 'confirmado_doble': si la base era 'media', sube a 'alta'
          (validación cruzada exitosa)
        - estado 'ml_no_confirma': sin cambio (la regex es alta confianza y
          el ML disiente, pero se prioriza la regex)
        - resto de estados: sin cambio

    Esta confiabilidad refleja la calidad técnica del procesamiento, NO
    la probabilidad clínica de que la alerta sea correcta.
    """
    confianza_birads = resultado_birads.get("confianza", "no_detectado")
    confianza_rec = resultado_recomendacion.get("confianza", "no_clasificada")
    metodo_rec = resultado_recomendacion.get("metodo")

    # Cálculo base (lógica original)
    if confianza_birads == "baja" or confianza_rec == "baja":
        nivel_base = "baja"
    elif confianza_birads == "media" or confianza_rec == "media":
        nivel_base = "media"
    elif metodo_rec == "tf_idf_similitud":
        nivel_base = "media"
    else:
        nivel_base = "alta"

    # Ajuste por verificación ML (si está disponible)
    if verificacion_ml is None:
        return nivel_base

    estado_ml = verificacion_ml.get("estado_verificacion")

    if estado_ml == "discrepante_real":
        # El ML detecta inconsistencia técnica real → bajar confiabilidad
        return "baja"

    if estado_ml == "confirmado_doble" and nivel_base == "media":
        # Validación cruzada exitosa sobre extracción dudosa → subir
        return "alta"

    # Estados que no modifican: confirmado, ml_no_confirma, ml_inseguro,
    # no_verificable
    return nivel_base


def _es_mas_urgente(categoria_a: str, categoria_b: str) -> bool:
    """Devuelve True si categoria_a es más urgente que categoria_b
    según la jerarquía clínica (posición más baja = más urgente).
    """
    try:
        return (
            JERARQUIA_URGENCIA.index(categoria_a)
            < JERARQUIA_URGENCIA.index(categoria_b)
        )
    except ValueError:
        return False


# Criticidad clínica de cada categoría BI-RADS: qué tan grave es que la
# recomendación se quede corta. Deriva del estándar ACR, no del corpus.
#   crítico  = sospecha de malignidad (un retraso puede ser grave)
#   medio    = estudio incompleto o malignidad ya en manejo
#   bajo     = hallazgo benigno o normal
_CRITICIDAD_BIRADS = {
    5: "critico", 4: "critico",
    0: "medio", 6: "medio",
    3: "bajo", 2: "bajo", 1: "bajo",
}

# Conductas que constituyen una acción diagnóstica/definitiva ante sospecha.
_ACCION_DIAGNOSTICA = {"biopsia_histologia", "derivacion_oncologica"}


def _nivel_por_impacto(birads, principal, esperada):
    """Clasifica una recomendación menos urgente que la esperada (incoherencia).
    La incoherencia siempre genera alerta; lo que varía es su severidad, según
    la criticidad del BI-RADS y cuánto se queda corta la recomendación.

    Returns: (estado, severidad, regla_aplicada)
        estado: siempre 'incoherente'
        severidad: 'baja' | 'media' | 'alta' | 'critica'
    """
    # Regla central (validada): sospecha de malignidad (BI-RADS 4/5) cuya
    # recomendación NO incluye acción diagnóstica -> siempre ALERTA CRÍTICA.
    if birads in (4, 5) and principal not in _ACCION_DIAGNOSTICA:
        return "incoherente", "critica", "regla_sospecha_sin_accion_diagnostica"

    # Magnitud: cuántos niveles de urgencia por debajo de lo esperado queda la
    # recomendación (en la jerarquía, índice mayor = menos urgente).
    try:
        gap = (JERARQUIA_URGENCIA.index(principal)
               - JERARQUIA_URGENCIA.index(esperada))
    except (ValueError, TypeError):
        gap = 1
    desvio_grande = gap >= 2

    criticidad = _CRITICIDAD_BIRADS.get(birads, "bajo")
    if criticidad == "critico":
        # (cubierto arriba, pero por completitud)
        severidad = "critica" if desvio_grande else "alta"
    elif criticidad == "medio":
        # incoherencia relevante -> alta; menor -> baja (sin riesgo)
        severidad = "alta" if desvio_grande else "baja"
    else:  # criticidad baja
        # incoherencia relevante -> media; menor -> baja (sin riesgo)
        severidad = "media" if desvio_grande else "baja"
    return "incoherente", severidad, "regla_recomendacion_insuficiente"


def _aplicar_reglas_cotejo(
    birads: int,
    categorias_detectadas: List[str],
) -> Dict[str, Any]:
    """Aplica las reglas de coherencia clínica paso a paso.

    Returns:
        dict con:
            - estado: 'coherente' | 'coherente_equivalente' |
                      'notificacion' | 'coherente_con_precaucion' |
                      'incoherente' | 'birads_desconocido'
            - severidad: heredada de la tabla ACR o ajustada a 'baja' si notificacion
            - regla_aplicada: cuál de los pasos resolvió el cotejo
            - detalle_verificacion: las 4 verificaciones realizadas
    """
    if birads not in TABLA_ACR:
        return {
            "estado": "birads_desconocido",
            "severidad": None,
            "regla_aplicada": None,
            "detalle_verificacion": {
                "birads_no_esta_en_tabla": True,
            },
        }

    config = TABLA_ACR[birads]
    esperada = config["esperada"]
    equivalentes = config["equivalentes_aceptables"]
    con_notificacion = config["equivalentes_con_notificacion"]
    severidad_base = config["severidad"]

    detalle = {
        "esperada_acr": esperada,
        "equivalentes_aceptables": equivalentes,
        "equivalentes_con_notificacion": con_notificacion,
        "esperada_presente": esperada in categorias_detectadas,
        "equivalente_presente": None,
        "notificacion_presente": None,
        "principal_es_mas_urgente": None,
    }

    # Regla 1: ¿la esperada está en las categorías detectadas?
    if esperada in categorias_detectadas:
        return {
            "estado": "coherente",
            "severidad": severidad_base,
            "regla_aplicada": "regla_1_esperada_presente",
            "detalle_verificacion": detalle,
        }

    # Regla 2: ¿alguna equivalente aceptable está?
    equivalente_match = next(
        (e for e in equivalentes if e in categorias_detectadas), None
    )
    detalle["equivalente_presente"] = equivalente_match
    if equivalente_match:
        return {
            "estado": "coherente_equivalente",
            "severidad": severidad_base,
            "regla_aplicada": "regla_2_equivalente_aceptable",
            "detalle_verificacion": detalle,
        }

    # Regla 3: ¿alguna equivalente con notificación está? -> INCOHERENCIA BAJA
    # (conducta que se queda corta, sin riesgo para la paciente: alerta leve).
    notificacion_match = next(
        (e for e in con_notificacion if e in categorias_detectadas), None
    )
    detalle["notificacion_presente"] = notificacion_match
    if notificacion_match:
        return {
            "estado": "incoherente",
            "severidad": "baja",
            "regla_aplicada": "regla_3_incoherencia_leve",
            "detalle_verificacion": detalle,
        }

    # Regla 4: ¿la principal detectada es más urgente que la esperada?
    if categorias_detectadas:
        unicas = set(categorias_detectadas)
        principal = next(
            (c for c in JERARQUIA_URGENCIA if c in unicas), None
        )
        if principal and _es_mas_urgente(principal, esperada):
            detalle["principal_es_mas_urgente"] = principal
            # Regla 4b (dirigida por la tabla): ciertas conductas más agresivas se
            # marcan para revisión según el campo 'marcar_revision' de la categoría.
            # Esto captura casos como un BI-RADS benigno con acción invasiva, sin
            # hardcodear la lógica: toda la decisión vive en la tabla ACR.
            marcar = config.get("marcar_revision", {})
            if principal in marcar:
                # Toda conducta marcada es una incoherencia; la severidad viene
                # de la tabla (baja, media, ...). La 'baja' es la alerta más leve.
                return {
                    "estado": "incoherente",
                    "severidad": marcar[principal],
                    "regla_aplicada": "regla_4b_marcar_revision",
                    "detalle_verificacion": detalle,
                }
            return {
                "estado": "coherente_con_precaucion",
                "severidad": severidad_base,
                "regla_aplicada": "regla_4_principal_mas_urgente",
                "detalle_verificacion": detalle,
            }

    # Ninguna regla previa aplica: la recomendación se queda corta frente a lo
    # esperado. El desenlace (ALERTA graduada u OBSERVACIÓN) se calcula por
    # impacto en salud, con regla nombrada para trazabilidad completa.
    principal_detectada = None
    if categorias_detectadas:
        unicas = set(categorias_detectadas)
        principal_detectada = next(
            (c for c in JERARQUIA_URGENCIA if c in unicas), None
        )
    estado_nivel, severidad_nivel, regla_nivel = _nivel_por_impacto(
        birads, principal_detectada, esperada
    )
    return {
        "estado": estado_nivel,
        "severidad": severidad_nivel,
        "regla_aplicada": regla_nivel,
        "detalle_verificacion": detalle,
    }


# =============================================================================
# FUNCIÓN PÚBLICA 1: cotejar_birads_vs_recomendacion
# =============================================================================

def cotejar_birads_vs_recomendacion(
    resultado_birads: Dict[str, Any],
    resultado_recomendacion: Dict[str, Any],
    verificacion_ml: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coteja BI-RADS declarado vs recomendación detectada según norma ACR.

    Esta es la función principal del módulo. Recibe los diccionarios que
    devuelven los dos extractores anteriores y aplica la lógica de cotejo
    en cuatro pasos:

        1. ¿La recomendación esperada por ACR está entre las detectadas?
        2. ¿Alguna equivalente clínicamente aceptable está?
        3. ¿Alguna equivalente con notificación suave está?
        4. ¿La principal detectada es más urgente que la esperada?

    Si ninguna regla aplica, se genera una alerta real con la severidad
    definida en la tabla ACR para ese BI-RADS.

    Opcionalmente recibe el resultado del verificador ML (verificacion_ml)
    para ajustar la confiabilidad técnica:
        - Si el ML detecta 'discrepante_real' → confiabilidad técnica 'baja'
        - Si el ML detecta 'confirmado_doble' sobre regex media → 'alta'
        - Resto de estados: sin cambio

    Args:
        resultado_birads: dict del extractor_birads. Debe tener al menos:
            'birads_conclusion', 'confianza', 'menciones_adicionales'.
        resultado_recomendacion: dict del extractor_recomendacion. Debe
            tener al menos: 'categorias_detectadas', 'categoria_principal',
            'confianza', 'metodo'.
        verificacion_ml: dict opcional del verificador_birads_ml. Si se pasa,
            debe tener al menos: 'estado_verificacion', 'birads_predicho_ml',
            'confianza_ml'.

    Returns:
        Dict con la decisión clínica y trazabilidad completa. Incluye el
        campo 'verificacion_ml' que refleja la información del verificador
        ML (None si no fue provista).
    """
    birads = resultado_birads.get("birads_conclusion")
    categorias = resultado_recomendacion.get("categorias_detectadas", [])
    categoria_principal = resultado_recomendacion.get("categoria_principal")
    menciones_adicionales = resultado_birads.get("menciones_adicionales", [])

    # Validación de inputs
    if birads is None:
        return {
            "estado": "no_procesable",
            "severidad": None,
            "requiere_alerta": False,
            "birads": None,
            "recomendacion_esperada": None,
            "recomendacion_detectada_principal": categoria_principal,
            "categorias_detectadas": categorias,
            "menciones_adicionales_birads": menciones_adicionales,
            "confiabilidad_tecnica": "no_aplicable",
            "verificacion_ml": verificacion_ml,
            "mensaje": (
                "No se pudo procesar el cotejo: el extractor de BI-RADS no "
                "detectó una categoría declarada en la conclusión del informe."
            ),
            "trazabilidad": {
                "razon_no_procesable": "birads_conclusion es None",
            },
        }

    # Chequeo de extracción: si no se detectó o no se pudo clasificar la
    # recomendación, el sistema no puede cotejar coherencia. En lugar de asumir
    # una incoherencia clínica, se marca REVISIÓN POR EXTRACCIÓN (abstención
    # honesta): puede ser una omisión clínica real o un formato que el sistema
    # no supo leer. Para BI-RADS de sospecha (4/5) la prioridad es alta, por si
    # se tratara de una omisión peligrosa.
    sin_recomendacion = (not categorias) or (categoria_principal in (None, "ambigua"))
    if sin_recomendacion:
        criticidad = _CRITICIDAD_BIRADS.get(birads, "bajo")
        prioridad = "alta" if criticidad == "critico" else "media"
        return {
            "estado": "revision_extraccion",
            "severidad": prioridad,
            "requiere_alerta": True,
            "birads": birads,
            "recomendacion_esperada": TABLA_ACR.get(birads, {}).get("esperada"),
            "recomendacion_detectada_principal": categoria_principal,
            "categorias_detectadas": categorias,
            "menciones_adicionales_birads": menciones_adicionales,
            "confiabilidad_tecnica": "baja",
            "verificacion_ml": verificacion_ml,
            "mensaje": (
                f"Revisión manual: no se pudo extraer ni clasificar una "
                f"recomendación para este informe (BI-RADS {birads}). Puede ser "
                f"una omisión de la recomendación o un formato no reconocido por "
                f"el sistema. No se emite juicio de coherencia; se recomienda "
                f"revisión."
            ),
            "trazabilidad": {
                "regla_aplicada": "revision_por_extraccion",
                "razon": "recomendacion_no_detectada_o_ambigua",
            },
        }

    # Aplicar reglas de cotejo
    resultado_reglas = _aplicar_reglas_cotejo(birads, categorias)

    # Calcular confiabilidad técnica (ahora considerando verificacion_ml)
    confiabilidad = _calcular_confiabilidad_tecnica(
        resultado_birads, resultado_recomendacion, verificacion_ml
    )

    # Determinar si requiere alerta
    requiere_alerta = resultado_reglas["estado"] == "incoherente"

    # Construir mensaje clínico
    if requiere_alerta:
        if resultado_reglas.get("regla_aplicada") == "regla_4b_marcar_revision":
            principal = resultado_reglas["detalle_verificacion"].get(
                "principal_es_mas_urgente", "una conducta más agresiva"
            )
            desc_cat = ("estudio sin hallazgos" if birads == 1
                        else "hallazgo benigno definitivo")
            mensaje = (
                f"BI-RADS {birads} ({desc_cat}) corresponde a control anual "
                f"rutinario según ACR, pero la recomendación detectada "
                f"('{principal}') implica una conducta más agresiva. Esta desviación "
                f"del protocolo puede indicar que la categoría BI-RADS o la "
                f"recomendación no reflejan bien los hallazgos. Se sugiere revisión."
            )
        elif resultado_reglas.get("severidad") == "baja":
            det = resultado_reglas["detalle_verificacion"]
            conducta = (det.get("notificacion_presente")
                        or det.get("principal_es_mas_urgente")
                        or "la conducta indicada")
            mensaje = (
                f"Incoherencia leve: en el BI-RADS {birads}, la recomendación "
                f"('{conducta}') se aparta de la esperada por ACR, pero de forma "
                "menor y sin riesgo aparente para la paciente. Alerta de baja "
                "prioridad; revisar cuando sea posible."
            )
        else:
            mensaje = MENSAJES_INCONSISTENCIA.get(
                birads, "Inconsistencia detectada entre BI-RADS y recomendación."
            )
    elif resultado_reglas["estado"] == "coherente_con_precaucion":
        mensaje = (
            f"Coherente con precaución: el radiólogo optó por una conducta "
            f"más conservadora ('{resultado_reglas['detalle_verificacion']['principal_es_mas_urgente']}') "
            f"que la mínima esperada por ACR para BI-RADS {birads} "
            f"('{resultado_reglas['detalle_verificacion']['esperada_acr']}')."
        )
    elif resultado_reglas["estado"] == "coherente_equivalente":
        mensaje = (
            f"Coherente: la recomendación detectada incluye una alternativa "
            f"clínicamente aceptable a la esperada para BI-RADS {birads}."
        )
    else:
        mensaje = f"Recomendación coherente con norma ACR para BI-RADS {birads}."

    return {
        "estado": resultado_reglas["estado"],
        "severidad": resultado_reglas["severidad"],
        "requiere_alerta": requiere_alerta,
        "birads": birads,
        "recomendacion_esperada": resultado_reglas["detalle_verificacion"].get(
            "esperada_acr"
        ),
        "recomendacion_detectada_principal": categoria_principal,
        "categorias_detectadas": categorias,
        "menciones_adicionales_birads": menciones_adicionales,
        "confiabilidad_tecnica": confiabilidad,
        "verificacion_ml": verificacion_ml,
        "mensaje": mensaje,
        "trazabilidad": {
            "regla_aplicada": resultado_reglas["regla_aplicada"],
            "detalle_verificacion": resultado_reglas["detalle_verificacion"],
            "extraccion_birads": {
                "valor": birads,
                "confianza": resultado_birads.get("confianza"),
                "fuente": resultado_birads.get("fuente"),
                "encabezado": resultado_birads.get("encabezado_conclusion"),
            },
            "extraccion_recomendacion": {
                "texto_original": resultado_recomendacion.get(
                    "trazabilidad", {}
                ).get("texto_original"),
                "texto_normalizado": resultado_recomendacion.get(
                    "trazabilidad", {}
                ).get("texto_normalizado"),
                "categorias": categorias,
                "principal": categoria_principal,
                "confianza": resultado_recomendacion.get("confianza"),
                "metodo": resultado_recomendacion.get("metodo"),
            },
            "verificacion_ml_resumen": (
                {
                    "estado": verificacion_ml.get("estado_verificacion"),
                    "birads_ml": verificacion_ml.get("birads_predicho_ml"),
                    "confianza_ml": verificacion_ml.get("confianza_ml"),
                    "regla_ml": verificacion_ml.get("regla_aplicada"),
                }
                if verificacion_ml
                else None
            ),
        },
    }


# =============================================================================
# FUNCIÓN PÚBLICA 2: generar_reporte_alerta_detallado (Mockup A)
# =============================================================================

def generar_reporte_alerta_detallado(
    resultado_cotejo: Dict[str, Any],
    informe_id: Optional[str] = None,
) -> str:
    """Genera el reporte detallado de alerta (Mockup A).

    Formato pensado para vista de detalle en el dashboard, cuando un
    revisor abre un caso específico para inspeccionarlo.

    Args:
        resultado_cotejo: dict devuelto por cotejar_birads_vs_recomendacion.
        informe_id: identificador opcional del informe.

    Returns:
        String con el reporte formateado.
    """
    sev_upper = (resultado_cotejo.get("severidad") or "").upper()
    conf_upper = (resultado_cotejo.get("confiabilidad_tecnica") or "").upper()
    linea = "═" * 75
    sublinea = "─" * 75

    lineas: List[str] = []
    lineas.append(linea)
    lineas.append(
        f"ALERTA CLÍNICA — Severidad: {sev_upper} — "
        f"Confiabilidad técnica: {conf_upper}"
    )
    lineas.append(linea)
    if informe_id:
        lineas.append(f"Informe ID:      {informe_id}")
    lineas.append(f"Procesado:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas.append(f"Estado cotejo:   {resultado_cotejo.get('estado', '').upper()}")

    # Sección: Resumen del caso
    lineas.append(f"\n{sublinea}")
    lineas.append("RESUMEN DEL CASO")
    lineas.append(sublinea)
    lineas.append(f"  BI-RADS declarado:           {resultado_cotejo['birads']}")
    lineas.append(
        f"  Recomendación esperada ACR:  {resultado_cotejo['recomendacion_esperada']}"
    )
    lineas.append(
        f"  Recomendación detectada:     "
        f"{resultado_cotejo['recomendacion_detectada_principal']}"
    )
    lineas.append("")
    lineas.append(f"  {resultado_cotejo['mensaje']}")

    # Menciones BI-RADS adicionales (contexto)
    menciones = resultado_cotejo.get("menciones_adicionales_birads", [])
    if menciones:
        lineas.append(f"\n  Contexto adicional:")
        lineas.append(
            f"    El informe contiene menciones de otras categorías BI-RADS "
            f"({menciones}) fuera del bloque CONCLUSIÓN. Esto no afecta la "
            f"decisión del cotejo pero puede indicar referencias a estudios "
            f"complementarios o previos."
        )

    # Sección: Confiabilidad técnica
    lineas.append(f"\n{sublinea}")
    lineas.append("CONFIABILIDAD TÉCNICA DEL PROCESAMIENTO")
    lineas.append(sublinea)
    lineas.append(f"  Nivel: {conf_upper}")
    lineas.append("")
    lineas.append(
        "  Significa: refleja la calidad técnica del procesamiento (qué tan\n"
        "  estrictas fueron las reglas aplicadas en cada paso).\n"
        "\n"
        "  Niveles posibles:\n"
        "    - alta:  todas las extracciones con regla estricta\n"
        "    - media: alguna extracción usó fallback semántico o typos\n"
        "    - baja:  alguna extracción con confianza reducida\n"
        "\n"
        "  Esta confiabilidad NO es una probabilidad clínica.\n"
        "  Toda alerta requiere validación humana."
    )

    # Sección: Verificación ML (si está presente)
    verif_ml = resultado_cotejo.get("verificacion_ml")
    if verif_ml:
        lineas.append(f"\n{sublinea}")
        lineas.append("VERIFICACIÓN DUAL DE EXTRACCIÓN (regex + ML)")
        lineas.append(sublinea)
        estado_ml = verif_ml.get("estado_verificacion", "desconocido")
        birads_ml = verif_ml.get("birads_predicho_ml")
        conf_ml = verif_ml.get("confianza_ml")
        lineas.append(f"  Estado:        {estado_ml}")
        lineas.append(f"  BI-RADS regex: {resultado_cotejo['birads']}")
        if birads_ml is not None:
            lineas.append(
                f"  BI-RADS ML:    {birads_ml} (confianza {conf_ml:.2f})"
            )
        if verif_ml.get("mensaje"):
            lineas.append("")
            lineas.append(f"  {verif_ml['mensaje']}")

    # Sección: Extracciones realizadas
    lineas.append(f"\n{sublinea}")
    lineas.append("EXTRACCIONES REALIZADAS")
    lineas.append(sublinea)
    trz = resultado_cotejo.get("trazabilidad", {})
    ex_bi = trz.get("extraccion_birads", {})
    ex_re = trz.get("extraccion_recomendacion", {})

    lineas.append(
        f"  BI-RADS:    {ex_bi.get('valor')} | "
        f"confianza: {ex_bi.get('confianza')} | "
        f"fuente: {ex_bi.get('fuente')}"
    )
    if ex_bi.get("encabezado"):
        lineas.append(f"              encabezado detectado: '{ex_bi['encabezado']}'")

    lineas.append("")
    lineas.append("  Recomendación:")
    lineas.append(f"    Texto original:    {ex_re.get('texto_original', '')[:120]}")
    lineas.append(f"    Texto normalizado: {ex_re.get('texto_normalizado', '')[:120]}")
    lineas.append(f"    Categorías:        {ex_re.get('categorias')}")
    lineas.append(f"    Principal:         {ex_re.get('principal')}")
    lineas.append(
        f"    Confianza:         {ex_re.get('confianza')} | "
        f"método: {ex_re.get('metodo')}"
    )

    # Sección: Cotejo contra norma ACR
    lineas.append(f"\n{sublinea}")
    lineas.append("COTEJO CONTRA NORMA ACR")
    lineas.append(sublinea)
    detalle = trz.get("detalle_verificacion", {})
    lineas.append(f"  Regla aplicada (BI-RADS {resultado_cotejo['birads']}):")
    lineas.append(f"    - Esperada:     {detalle.get('esperada_acr')}")
    lineas.append(
        f"    - Equivalentes: {detalle.get('equivalentes_aceptables') or 'ninguno'}"
    )
    lineas.append(f"    - Severidad:    {resultado_cotejo.get('severidad')}")
    lineas.append("")
    lineas.append("  Verificación paso a paso:")
    check_esperada = "✓" if detalle.get("esperada_presente") else "✗"
    check_equiv = "✓" if detalle.get("equivalente_presente") else "✗"
    check_notif = "✓" if detalle.get("notificacion_presente") else "✗"
    check_urg = "✓" if detalle.get("principal_es_mas_urgente") else "✗"
    lineas.append(
        f"    [{check_esperada}] ¿Esperada ACR está entre las detectadas?"
    )
    lineas.append(f"    [{check_equiv}] ¿Alguna equivalente aceptable está?")
    lineas.append(f"    [{check_notif}] ¿Alguna equivalente con notificación está?")
    lineas.append(f"    [{check_urg}] ¿Principal detectada es más urgente que esperada?")
    lineas.append("")
    lineas.append(f"  Regla resolutoria: {trz.get('regla_aplicada')}")

    # Sección: Sugerencias genéricas (solo si requiere alerta)
    if resultado_cotejo.get("requiere_alerta"):
        lineas.append(f"\n{sublinea}")
        lineas.append("SUGERENCIAS GENÉRICAS PARA EL REVISOR")
        lineas.append(sublinea)
        for sug in SUGERENCIAS_GENERICAS:
            lineas.append(f"  • {sug}")

    lineas.append(f"\n{linea}")
    return "\n".join(lineas)


# =============================================================================
# FUNCIÓN PÚBLICA 3: generar_reporte_alerta_compacto (Mockup B)
# =============================================================================

def generar_reporte_alerta_compacto(
    resultado_cotejo: Dict[str, Any],
    informe_id: Optional[str] = None,
) -> str:
    """Genera el reporte compacto de alerta (Mockup B).

    Formato pensado para listas y tablas (vista resumen del dashboard).
    Una sola alerta debe caber en pocas líneas.

    Args:
        resultado_cotejo: dict devuelto por cotejar_birads_vs_recomendacion.
        informe_id: identificador opcional.

    Returns:
        String compacto de la alerta.
    """
    sev = (resultado_cotejo.get("severidad") or "").upper()
    birads = resultado_cotejo.get("birads")
    esperada = resultado_cotejo.get("recomendacion_esperada")
    detectada = resultado_cotejo.get("recomendacion_detectada_principal")
    id_str = informe_id if informe_id else "?"

    trz = resultado_cotejo.get("trazabilidad", {})
    texto = trz.get("extraccion_recomendacion", {}).get("texto_original", "")[:100]

    lineas = [
        f"[{sev}] {id_str} | BI-RADS {birads} → esperado '{esperada}', detectado '{detectada}'",
        f"  Texto: \"{texto}\"",
        f"  {resultado_cotejo.get('mensaje', '')[:200]}",
    ]
    return "\n".join(lineas)


# =============================================================================
# FUNCIÓN PÚBLICA 4: resumen_para_dataframe (Mockup C)
# =============================================================================

def resumen_para_dataframe(
    lista_resultados_cotejo: List[Dict[str, Any]],
    lista_informe_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Construye una representación tabular de múltiples cotejos.

    Útil para exportar a CSV o cargar en pandas DataFrame en el dashboard.

    Args:
        lista_resultados_cotejo: lista de dicts devueltos por cotejar_*.
        lista_informe_ids: identificadores correspondientes (mismo orden).

    Returns:
        Lista de dicts, cada uno representando una fila del CSV.
    """
    filas = []
    for i, resultado in enumerate(lista_resultados_cotejo):
        informe_id = (
            lista_informe_ids[i]
            if lista_informe_ids and i < len(lista_informe_ids)
            else f"informe_{i}"
        )
        trz = resultado.get("trazabilidad", {})
        ex_re = trz.get("extraccion_recomendacion", {})
        verif_ml = resultado.get("verificacion_ml")

        filas.append({
            "informe_id": informe_id,
            "birads": resultado.get("birads"),
            "estado": resultado.get("estado"),
            "severidad": resultado.get("severidad"),
            "requiere_alerta": resultado.get("requiere_alerta"),
            "confiabilidad_tecnica": resultado.get("confiabilidad_tecnica"),
            "recomendacion_esperada": resultado.get("recomendacion_esperada"),
            "recomendacion_detectada": resultado.get(
                "recomendacion_detectada_principal"
            ),
            "categorias_detectadas": str(resultado.get("categorias_detectadas")),
            "texto_recomendacion": ex_re.get("texto_original", "")[:200],
            "verificacion_ml_estado": (
                verif_ml.get("estado_verificacion") if verif_ml else None
            ),
            "verificacion_ml_birads": (
                verif_ml.get("birads_predicho_ml") if verif_ml else None
            ),
            "mensaje": resultado.get("mensaje", "")[:300],
        })
    return filas


# =============================================================================
# FUNCIÓN PÚBLICA 5: crear_resumen_compacto (estadísticas agregadas)
# =============================================================================

def crear_resumen_compacto(
    lista_resultados_cotejo: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Genera estadísticas agregadas de un conjunto de cotejos.

    Pensado para alimentar el dashboard con métricas resumidas y la
    lista de casos urgentes ordenada por severidad.

    Args:
        lista_resultados_cotejo: lista de dicts devueltos por cotejar_*.

    Returns:
        Dict con métricas y casos top.
    """
    total = len(lista_resultados_cotejo)
    if total == 0:
        return {
            "total_procesados": 0,
            "alertas_total": 0,
            "por_estado": {},
            "por_severidad": {},
            "por_birads": {},
            "casos_urgentes_top": [],
        }

    # Conteo por estado
    por_estado: Dict[str, int] = {}
    por_severidad: Dict[str, int] = {}
    por_birads: Dict[int, int] = {}
    alertas = []

    for resultado in lista_resultados_cotejo:
        estado = resultado.get("estado", "desconocido")
        severidad = resultado.get("severidad")
        birads = resultado.get("birads")

        por_estado[estado] = por_estado.get(estado, 0) + 1
        if severidad:
            por_severidad[severidad] = por_severidad.get(severidad, 0) + 1
        if birads is not None:
            por_birads[birads] = por_birads.get(birads, 0) + 1

        if resultado.get("requiere_alerta"):
            alertas.append(resultado)

    # Ordenar alertas por severidad (crítica > alta > media > baja)
    orden_severidad = {"critica": 0, "alta": 1, "media": 2, "baja": 3}
    alertas_ordenadas = sorted(
        alertas,
        key=lambda r: orden_severidad.get(r.get("severidad", "baja"), 99),
    )

    return {
        "total_procesados": total,
        "alertas_total": len(alertas),
        "tasa_alertas_pct": round(100 * len(alertas) / total, 2),
        "por_estado": por_estado,
        "por_severidad": por_severidad,
        "por_birads": dict(sorted(por_birads.items())),
        "casos_urgentes_top": alertas_ordenadas[:50],
    }


# =============================================================================
# FUNCIÓN HELPER: guardar_alerta_json
# =============================================================================

def guardar_alerta_json(
    resultado_cotejo: Dict[str, Any],
    informe_id: str,
    output_dir: str = "audit_logs",
) -> str:
    """Persiste un resultado de cotejo como JSON con timestamp.

    Args:
        resultado_cotejo: dict de cotejar_birads_vs_recomendacion.
        informe_id: identificador del informe.
        output_dir: directorio destino.

    Returns:
        Ruta del archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"cotejo_{informe_id}_{timestamp}.json"
    ruta = os.path.join(output_dir, nombre)

    payload = {
        "informe_id": informe_id,
        "timestamp": datetime.now().isoformat(),
        "resultado_cotejo": resultado_cotejo,
    }

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    return ruta


# =============================================================================
# TESTS INLINE
# =============================================================================

def _ejecutar_tests() -> None:
    """Suite de tests inline. Ejecutar con: python -m src.cotejo_acr"""

    # Caso 1: BI-RADS 2 + control anual = COHERENTE
    r_birads = {
        "birads_conclusion": 2,
        "confianza": "alta",
        "fuente": "bloque_conclusion_estricto",
        "menciones_adicionales": [],
    }
    r_rec = {
        "categorias_detectadas": ["control_anual"],
        "categoria_principal": "control_anual",
        "confianza": "alta",
        "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere control mamográfico anual.",
            "texto_normalizado": "- se sugiere control mamografico anual.",
        },
    }
    res = cotejar_birads_vs_recomendacion(r_birads, r_rec)

    casos = []

    casos.append({
        "nombre": "C1: BI-RADS 2 + control_anual → coherente",
        "resultado": res,
        "esperado": {"estado": "coherente", "requiere_alerta": False},
    })

    # Caso 2: BI-RADS 4 sin biopsia = ALERTA ALTA
    r_birads = {"birads_conclusion": 4, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["criterio_medico"],
        "categoria_principal": "criterio_medico",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Controles según criterio del médico tratante.",
            "texto_normalizado": "- controles segun criterio del medico tratante.",
        },
    }
    casos.append({
        "nombre": "C2: BI-RADS 4 + criterio_medico → alerta CRÍTICA (sospecha sin acción diagnóstica)",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "incoherente", "severidad": "critica", "requiere_alerta": True},
    })

    # Caso 3: BI-RADS 5 sin biopsia = ALERTA CRÍTICA
    r_birads = {"birads_conclusion": 5, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["estudio_complementario_imagen"],
        "categoria_principal": "estudio_complementario_imagen",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere ecografía mamaria para posterior recategorización.",
            "texto_normalizado": "- se sugiere ecografia mamaria para posterior recategorizacion.",
        },
    }
    casos.append({
        "nombre": "C3: BI-RADS 5 + estudio_complementario → alerta CRÍTICA",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "incoherente", "severidad": "critica", "requiere_alerta": True},
    })

    # Caso 4: BI-RADS 0 + correlacion eco = COHERENTE EQUIVALENTE
    r_birads = {"birads_conclusion": 0, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["correlacion_ecografica"],
        "categoria_principal": "correlacion_ecografica",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere correlación con ecografía mamaria.",
            "texto_normalizado": "- se sugiere correlacion con ecografia mamaria.",
        },
    }
    casos.append({
        "nombre": "C4: BI-RADS 0 + correlacion_ecografica → coherente_equivalente",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "coherente_equivalente", "requiere_alerta": False},
    })

    # Caso 5: BI-RADS 2 + control_corto_plazo = NOTIFICACIÓN suave
    r_birads = {"birads_conclusion": 2, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["control_corto_plazo"],
        "categoria_principal": "control_corto_plazo",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere control semestral.",
            "texto_normalizado": "- se sugiere control semestral.",
        },
    }
    casos.append({
        "nombre": "C5: BI-RADS 2 + control_corto_plazo → incoherencia leve (alerta baja)",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "incoherente", "severidad": "baja", "requiere_alerta": True},
    })

    # Caso 6: BI-RADS 3 + biopsia = COHERENTE EQUIVALENTE (más agresivo)
    r_birads = {"birads_conclusion": 3, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["biopsia_histologia"],
        "categoria_principal": "biopsia_histologia",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere biopsia.",
            "texto_normalizado": "- se sugiere biopsia.",
        },
    }
    casos.append({
        "nombre": "C6: BI-RADS 3 + biopsia → coherente_equivalente",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "coherente_equivalente", "requiere_alerta": False},
    })

    # Caso 7: BI-RADS 0 + control_corto_plazo = INCOHERENTE (no equivalente)
    r_birads = {"birads_conclusion": 0, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["control_corto_plazo"],
        "categoria_principal": "control_corto_plazo",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere control ecográfico semestral.",
            "texto_normalizado": "- se sugiere control ecografico semestral.",
        },
    }
    casos.append({
        "nombre": "C7: BI-RADS 0 + control_corto_plazo → incoherente (no aceptable)",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "incoherente", "severidad": "alta", "requiere_alerta": True},
    })

    # Caso 8: BI-RADS 1 + control anual + correlacion = COHERENTE (principal está)
    r_birads = {"birads_conclusion": 1, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["correlacion_ecografica", "control_anual"],
        "categoria_principal": "correlacion_ecografica",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Correlación con ecografía y control anual.",
            "texto_normalizado": "- correlacion con ecografia y control anual.",
        },
    }
    casos.append({
        "nombre": "C8: BI-RADS 1 + [correlacion, control_anual] → coherente",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "coherente", "requiere_alerta": False},
    })

    # Caso 9: Confiabilidad técnica MEDIA por TF-IDF
    r_birads = {"birads_conclusion": 2, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["control_anual"],
        "categoria_principal": "control_anual",
        "confianza": "media", "metodo": "tf_idf_similitud",
        "trazabilidad": {"texto_original": "...", "texto_normalizado": "..."},
    }
    casos.append({
        "nombre": "C9: Confiabilidad técnica = media por TF-IDF",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"confiabilidad_tecnica": "media"},
    })

    # Caso 10: BI-RADS no procesable
    r_birads = {"birads_conclusion": None, "confianza": "no_detectado",
                "fuente": None, "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": [],
        "categoria_principal": None,
        "confianza": "no_clasificada", "metodo": None,
        "trazabilidad": {"texto_original": "", "texto_normalizado": ""},
    }
    casos.append({
        "nombre": "C10: BI-RADS no detectado → no procesable",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec),
        "esperado": {"estado": "no_procesable", "requiere_alerta": False},
    })

    # =========================================================================
    # NUEVOS TESTS: Integración con verificador ML
    # =========================================================================

    # Caso 11: BI-RADS 2 + verificacion_ml=None → comportamiento idéntico a C1
    r_birads = {"birads_conclusion": 2, "confianza": "alta",
                "fuente": "bloque_conclusion_estricto", "menciones_adicionales": []}
    r_rec = {
        "categorias_detectadas": ["control_anual"],
        "categoria_principal": "control_anual",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere control mamográfico anual.",
            "texto_normalizado": "- se sugiere control mamografico anual.",
        },
    }
    casos.append({
        "nombre": "C11: verificacion_ml=None → comportamiento sin cambios",
        "resultado": cotejar_birads_vs_recomendacion(r_birads, r_rec, verificacion_ml=None),
        "esperado": {
            "estado": "coherente",
            "confiabilidad_tecnica": "alta",
            "verificacion_ml": None,
        },
    })

    # Caso 12: ML confirma → confiabilidad sin cambio
    verif_ml_confirma = {
        "estado_verificacion": "confirmado",
        "birads_predicho_ml": 2,
        "confianza_ml": 0.98,
        "regla_aplicada": "regla_1_doble_confirmacion_alta",
        "mensaje": "Regex y ML coinciden.",
    }
    casos.append({
        "nombre": "C12: ML confirma (alta) → confiabilidad sin cambio",
        "resultado": cotejar_birads_vs_recomendacion(
            r_birads, r_rec, verificacion_ml=verif_ml_confirma
        ),
        "esperado": {"confiabilidad_tecnica": "alta"},
    })

    # Caso 13: ML detecta discrepante_real → confiabilidad baja
    r_birads_baja = {"birads_conclusion": 4, "confianza": "baja",
                     "fuente": "bloque_conclusion_typos", "menciones_adicionales": []}
    verif_ml_discrep = {
        "estado_verificacion": "discrepante_real",
        "birads_predicho_ml": 0,
        "confianza_ml": 0.66,
        "regla_aplicada": "regla_6_discrepancia_real",
        "mensaje": "Texto atípico, revisar manualmente.",
    }
    casos.append({
        "nombre": "C13: ML discrepante_real → confiabilidad fuerza a baja",
        "resultado": cotejar_birads_vs_recomendacion(
            r_birads_baja, r_rec, verificacion_ml=verif_ml_discrep
        ),
        "esperado": {"confiabilidad_tecnica": "baja"},
    })

    # Caso 14: ML confirmado_doble sobre regex media → confiabilidad sube a alta
    r_birads_media = {"birads_conclusion": 0, "confianza": "media",
                      "fuente": "bloque_conclusion_typos", "menciones_adicionales": []}
    r_rec_estudio = {
        "categorias_detectadas": ["estudio_complementario_imagen"],
        "categoria_principal": "estudio_complementario_imagen",
        "confianza": "alta", "metodo": "regex",
        "trazabilidad": {
            "texto_original": "- Se sugiere ecografía complementaria.",
            "texto_normalizado": "- se sugiere ecografia complementaria.",
        },
    }
    verif_ml_doble = {
        "estado_verificacion": "confirmado_doble",
        "birads_predicho_ml": 0,
        "confianza_ml": 0.998,
        "regla_aplicada": "regla_5_validacion_cruzada",
        "mensaje": "Validación cruzada exitosa.",
    }
    casos.append({
        "nombre": "C14: ML confirmado_doble sobre regex media → confiabilidad sube a alta",
        "resultado": cotejar_birads_vs_recomendacion(
            r_birads_media, r_rec_estudio, verificacion_ml=verif_ml_doble
        ),
        "esperado": {"confiabilidad_tecnica": "alta"},
    })

    # Ejecutar verificaciones
    print("=" * 75)
    print("TESTS DE src/cotejo_acr.py")
    print("=" * 75)

    n_pasados = 0
    for caso in casos:
        resultado = caso["resultado"]
        esperado = caso["esperado"]
        checks = []
        for clave, valor_esperado in esperado.items():
            valor_real = resultado.get(clave)
            checks.append((
                valor_real == valor_esperado,
                f"{clave}: esperado={valor_esperado}, obtenido={valor_real}",
            ))

        paso = all(ok for ok, _ in checks)
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
        print("Estado: OK — cotejo_acr listo para uso en producción.")
    else:
        print("Estado: FALLA — revisar los casos que no pasaron.")


if __name__ == "__main__":
    _ejecutar_tests()
