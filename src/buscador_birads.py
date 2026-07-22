"""
buscador_birads.py — Búsqueda híbrida de la categoría BI-RADS final del informe.

Módulo del proyecto BME513 (Universidad de Valparaíso).

Resuelve el problema de encontrar el BI-RADS correcto en informes con
estructura heterogénea, donde no siempre existe un encabezado explícito
de "CONCLUSIÓN" y donde el término BI-RADS puede aparecer múltiples veces
(hallazgos, comparaciones históricas, conclusión final).

Estrategia: búsqueda híbrida en 4 fases

    FASE 1 — Regex exhaustiva
        Encuentra TODAS las menciones BI-RADS en el informe completo,
        no solo en el bloque conclusión.

    FASE 2 — Filtrado contextual
        Excluye menciones cuyo contexto previo indica que son referencias
        históricas, comparativas o educacionales (no la conclusión actual).

    FASE 3 — Ponderación posicional
        Otorga mayor score a menciones ubicadas en el último tercio del
        informe, donde clínicamente se ubica la conclusión.

    FASE 4 — Ranking ML (opcional, solo si hay empate)
        Cuando quedan varios candidatos con score máximo empatado,
        DistilBETO decide cuál mención es la conclusión final.

Filosofía Human-on-the-Loop:
- La regex es la AUTORIDAD CLÍNICA (extrae el número literal del informe).
- El ML solo ACTÚA COMO ÁRBITRO en casos de ambigüedad real.
- Cada decisión es AUDITABLE (trazabilidad explícita de cada fase).

Uso como librería:

    from src.buscador_birads import buscar_birads_final
    resultado = buscar_birads_final(full_report="...")

Uso como CLI:

    python -m src.buscador_birads              # ejecuta tests inline
    python -m src.buscador_birads --input informe.txt

Autor: Sebastián Inostroza Hurtado
Fecha: Junio 2026
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# VOCABULARIOS DE EXCLUSIÓN
# =============================================================================

# Palabras que indican referencia histórica (no es la conclusión actual)
PALABRAS_HISTORICAS = [
    "anterior", "anteriormente", "previo", "previa", "previos", "previas",
    "año pasado", "año anterior", "meses atrás", "hace",
    "histórico", "histórica", "historicos", "históricas",
    "estudio previo", "estudios previos", "informe previo", "informes previos",
    "mamografía previa", "mamografías previas",
    "mamografia previa", "mamografias previas",
    "ecografía previa", "ecografías previas",
    "ecografia previa", "ecografias previas",
    "control de", "seguimiento de",
]

# Palabras que indican comparación con otro estudio
PALABRAS_COMPARATIVAS = [
    "comparado con", "comparada con", "comparados con", "comparadas con",
    "en comparación", "en relación a", "en relación con",
    "respecto a", "respecto de", "respecto al",
    "vs.", "vs", "versus", "contra",
]

# Palabras que indican mención educacional o de sistema (no un caso concreto)
PALABRAS_EDUCACIONALES = [
    "sistema bi-rads", "categorías bi-rads", "clasificación bi-rads",
    "escala bi-rads", "según el sistema", "según la clasificación",
    "es decir", "por ejemplo", "por definición",
]

# Patrones que indican que la mención BI-RADS está NEGADA (no es la categoría
# asignada, sino una categoría que el radiólogo descarta). Ej: "los hallazgos
# no son sugerentes de BI-RADS 4".
#
# Se usan como REGEX (no substring) para tolerar palabras intermedias entre el
# disparador de negación y el concepto ("no SON sugerentes"), al estilo del
# algoritmo NegEx del PLN clínico.
#
# Criterio CONSERVADOR (alta precisión): solo patrones que niegan claramente una
# ASERCIÓN de categoría. No se dispara con negaciones de hallazgos ("no se
# observa nódulo") ni con "descartar" genérico ("descartar patología"), para no
# eliminar por error un BI-RADS válido.
PATRONES_NEGACION = [
    r"\bno\s+(?:\w+\s+){0,2}(?:corresponde|corresponden|sugerente|sugerentes|compatible|concordante)\b",
    r"\bno\s+cumple\s+criterios\b",
    r"\bsin\s+criterios\s+de\b",
    r"\bse\s+descarta\s+(?:que\s+)?(?:corresponda|sea|se\s+trate)\b",
]
_PATRONES_NEGACION_RE = [re.compile(p, re.IGNORECASE) for p in PATRONES_NEGACION]


# =============================================================================
# VOCABULARIOS PARA DETECCIÓN DE OMISIÓN (FASE 4)
# =============================================================================
# Se usan SOLO cuando no se encontró un BI-RADS válido, para distinguir:
#   (a) un informe sustantivo al que le FALTA la categoría  -> OMISIÓN (alerta)
#   (b) un fragmento trivial / sin contenido diagnóstico     -> no procesable
#
# Nota: todo el matching se hace sobre texto normalizado (sin tildes, minúsculas),
# por eso los términos van sin acento.

# Descriptores de hallazgos radiológicos (sustantivos que denotan contenido
# diagnóstico reportable).
HALLAZGOS_DESCRIPTORES = [
    "nodulo", "masa", "microcalcificaci", "calcificaci",
    "lesion", "realce", "distorsion", "asimetria",
    "ectasia", "adenopatia", "espiculad", "sospechos",
    "papilar", "quistic", "dilatacion ductal", "foco de realce",
]

# Acciones o recomendaciones clínicas que implican que el estudio es un informe
# diagnóstico real (y por tanto debía llevar categoría BI-RADS).
ACCIONES_CLINICAS = [
    "se sugiere", "se recomienda", "se aconseja", "sugiere realizar",
    "biopsia", "histolog", "puncion", "correlacion", "recategoriz",
    "control", "seguimiento", "derivar", "derivacion", "estudio complementario",
]

# Términos que elevan la severidad de la omisión (hallazgo potencialmente
# sospechoso sin categoría asignada).
TERMINOS_SEVERIDAD_ALTA = [
    "sospechos", "espiculad", "biopsia", "histolog", "malign", "papilar",
]


def _normalizar_para_deteccion(texto: str) -> str:
    """Minúsculas + sin tildes, para matching robusto del detector de omisión."""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.lower()


def detectar_hallazgos_reportables(full_report: str) -> Optional[Dict[str, Any]]:
    """FASE 4 (parte A): decide si un informe SIN BI-RADS es una omisión.

    Un informe se considera una OMISIÓN (y no un fragmento trivial) cuando
    contiene descriptores de hallazgos radiológicos y/o acciones clínicas que
    solo aparecen en un informe diagnóstico real.

    Returns:
        None si no hay evidencia de contenido reportable (no es omisión).
        Dict con la evidencia y la severidad sugerida si SÍ lo es.
    """
    texto = _normalizar_para_deteccion(full_report)

    hallazgos_encontrados = [h for h in HALLAZGOS_DESCRIPTORES if h in texto]
    acciones_encontradas = [a for a in ACCIONES_CLINICAS if a in texto]

    # Sin ninguna señal -> no es un informe diagnóstico sustantivo.
    if not hallazgos_encontrados and not acciones_encontradas:
        return None

    severidad = "alta" if any(t in texto for t in TERMINOS_SEVERIDAD_ALTA) else "media"

    return {
        "hallazgos_detectados": hallazgos_encontrados,
        "acciones_detectadas": acciones_encontradas,
        "severidad": severidad,
    }


# =============================================================================
# ESTRUCTURA DE DATOS PARA MENCIONES
# =============================================================================

@dataclass
class MencionBirads:
    """Representa una única mención de BI-RADS en el informe."""
    valor: int
    posicion_inicio: int
    posicion_fin: int
    texto_capturado: str
    contexto_previo: str = ""
    contexto_posterior: str = ""
    posicion_relativa: float = 0.0    # 0.0 = inicio, 1.0 = final
    score_posicional: int = 0
    descartada: bool = False
    razon_descarte: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valor": self.valor,
            "posicion_inicio": self.posicion_inicio,
            "posicion_fin": self.posicion_fin,
            "texto_capturado": self.texto_capturado,
            "contexto_previo": self.contexto_previo.strip(),
            "posicion_relativa": round(self.posicion_relativa, 3),
            "score_posicional": self.score_posicional,
            "descartada": self.descartada,
            "razon_descarte": self.razon_descarte,
        }


# =============================================================================
# FASE 1 — REGEX EXHAUSTIVA
# =============================================================================

# Patrones para capturar menciones BI-RADS con variaciones tipográficas.
# (?:US|MG|...) captura una etiqueta de modalidad opcional entre 'BI-RADS' y el
# número (p. ej. "BI-RADS US 2", "BI-RADS MG 4"), frecuente en informes reales.
_MODALIDAD = r"(?:US|MG|MMG|RM|RMN|MRI|ECO|TC|ECOGRAF[IÍ]A|MAMOGRAF[IÍ]A)?"
PATRONES_BIRADS = [
    # BI-RADS X, BIRADS X, BI RADS X (con o sin guión/espacios/modalidad)
    r"BI[\s\-]?RADS[®\s\-]*" + _MODALIDAD + r"[\s\-]*:?\s*(?P<valor>[0-6])",
    # BR X (abreviatura corta)
    r"\bBR[\s\-]*:?\s*(?P<valor>[0-6])\b",
    # Categoría BI-RADS X, Categoría X
    r"[Cc]ategor[íi]a[\s\-]*(?:BI[\s\-]?RADS)?[®\s\-]*" + _MODALIDAD + r"[\s\-]*:?\s*(?P<valor>[0-6])",
    # ACR X (mención por ACR)
    r"\bACR[\s\-]*:?\s*(?P<valor>[0-6])\b",
    # --- Variantes con números ROMANOS (frecuentes en informes reales) ---
    # BI-RADS II, BIRADS US IV, Birads -us II, etc. (0 se escribe como '0')
    r"BI[\s\-]?RADS[®\s\-]*" + _MODALIDAD + r"[\s\-]*:?\s*(?P<romano>VI|IV|V|III|II|I|0)\b",
    # Categoría BI-RADS III / Categoría IV
    r"[Cc]ategor[íi]a[\s\-]*(?:BI[\s\-]?RADS)?[®\s\-]*" + _MODALIDAD + r"[\s\-]*:?\s*(?P<romano>VI|IV|V|III|II|I|0)\b",
]

# Mapa de números romanos a arábigos para BI-RADS (0-6).
_ROMANO_A_ARABIGO = {
    "0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
}


def buscar_todas_menciones_birads(texto: str) -> List[MencionBirads]:
    """FASE 1: Busca todas las menciones BI-RADS en el texto completo.

    No aplica filtros de contexto ni posición. Solo extrae menciones brutas.
    Devuelve la lista ordenada por posición en el texto.
    """
    menciones: List[MencionBirads] = []
    posiciones_capturadas = set()

    for patron in PATRONES_BIRADS:
        for match in re.finditer(patron, texto, re.IGNORECASE):
            inicio = match.start()

            # Evitar duplicar la misma mención capturada por distintos patrones
            if inicio in posiciones_capturadas:
                continue

            # El valor puede venir como dígito arábigo ('valor') o como
            # número romano ('romano'), según el patrón que hizo match.
            grupos = match.groupdict()
            if grupos.get("valor") is not None:
                valor = int(grupos["valor"])
            elif grupos.get("romano") is not None:
                clave = grupos["romano"].upper()
                if clave not in _ROMANO_A_ARABIGO:
                    continue
                valor = _ROMANO_A_ARABIGO[clave]
            else:
                continue

            posiciones_capturadas.add(inicio)
            fin = match.end()
            texto_capturado = match.group(0)

            # Extraer contextos (50 caracteres antes y 50 después)
            contexto_previo = texto[max(0, inicio - 50):inicio]
            contexto_posterior = texto[fin:min(len(texto), fin + 50)]

            menciones.append(MencionBirads(
                valor=valor,
                posicion_inicio=inicio,
                posicion_fin=fin,
                texto_capturado=texto_capturado,
                contexto_previo=contexto_previo,
                contexto_posterior=contexto_posterior,
            ))

    # Ordenar por posición en el texto
    menciones.sort(key=lambda m: m.posicion_inicio)
    return menciones


# =============================================================================
# FASE 2 — FILTRADO CONTEXTUAL
# =============================================================================

def es_mencion_historica(contexto_previo: str) -> Optional[str]:
    """Detecta si el contexto previo indica una mención histórica.

    Devuelve el motivo del descarte, o None si no aplica.
    """
    contexto_lower = contexto_previo.lower()

    for palabra in PALABRAS_HISTORICAS:
        if palabra.lower() in contexto_lower:
            return f"contexto_historico:'{palabra}'"

    return None


def es_mencion_comparativa(contexto_previo: str) -> Optional[str]:
    """Detecta si el contexto previo indica una comparación con otro estudio."""
    contexto_lower = contexto_previo.lower()

    for palabra in PALABRAS_COMPARATIVAS:
        if palabra.lower() in contexto_lower:
            return f"contexto_comparativo:'{palabra}'"

    return None


def es_mencion_educacional(contexto_previo: str) -> Optional[str]:
    """Detecta si la mención es educacional o de sistema (no clínica concreta)."""
    contexto_lower = contexto_previo.lower()

    for frase in PALABRAS_EDUCACIONALES:
        if frase.lower() in contexto_lower:
            return f"contexto_educacional:'{frase}'"

    return None


def es_mencion_negada(contexto_previo: str) -> Optional[str]:
    """Detecta si la mención BI-RADS está negada (categoría descartada).

    Usa patrones regex tolerantes a palabras intermedias (estilo NegEx).
    Devuelve el motivo del descarte, o None si no aplica.
    """
    for patron in _PATRONES_NEGACION_RE:
        m = patron.search(contexto_previo)
        if m:
            return f"contexto_negado:'{m.group(0).strip()}'"

    return None


def filtrar_menciones_historicas(
    menciones: List[MencionBirads],
) -> List[MencionBirads]:
    """FASE 2: Marca como descartadas las menciones que NO son la conclusión.

    Descarta menciones históricas, comparativas, educacionales o NEGADAS.
    Modifica las menciones in-place agregando `descartada=True` y
    `razon_descarte`. Devuelve la lista completa (no filtrada).
    """
    for mencion in menciones:
        # Chequear cada tipo de exclusión
        razon = (
            es_mencion_historica(mencion.contexto_previo)
            or es_mencion_comparativa(mencion.contexto_previo)
            or es_mencion_educacional(mencion.contexto_previo)
            or es_mencion_negada(mencion.contexto_previo)
        )

        if razon:
            mencion.descartada = True
            mencion.razon_descarte = razon

    return menciones


# =============================================================================
# FASE 3 — PONDERACIÓN POSICIONAL
# =============================================================================

def calcular_score_posicional(posicion_relativa: float) -> int:
    """Score según la ubicación relativa en el texto.

    La conclusión clínica de un informe radiológico casi siempre está
    en el último 20-30% del texto.
    """
    if posicion_relativa >= 0.8:
        return 100  # último 20% — muy probablemente conclusión
    elif posicion_relativa >= 0.6:
        return 80   # 60-80% — probable conclusión
    elif posicion_relativa >= 0.4:
        return 40   # 40-60% — poco probable
    else:
        return 10   # primer 40% — muy poco probable (probablemente hallazgos)


def ponderar_por_posicion(
    menciones: List[MencionBirads],
    longitud_total: int,
) -> List[MencionBirads]:
    """FASE 3: Calcula posición relativa y score para cada mención."""
    if longitud_total == 0:
        return menciones

    for mencion in menciones:
        mencion.posicion_relativa = mencion.posicion_inicio / longitud_total
        mencion.score_posicional = calcular_score_posicional(
            mencion.posicion_relativa
        )

    return menciones


# =============================================================================
# FASE 4 — RANKING ML (SOLO SI HAY EMPATE)
# =============================================================================

def rankear_con_ml(
    candidatos: List[MencionBirads],
    texto_completo: str,
) -> tuple[MencionBirads, str]:
    """FASE 4: ML decide cuál candidato es la conclusión final.

    Solo se activa cuando hay 2+ candidatos con score máximo empatado.
    El ML analiza el contexto de cada candidato y elige el más probable.

    Devuelve la mención ganadora y el motivo de la elección.

    NOTA: Esta función asume que se importa DistilBETO desde el módulo
    verificador_birads_ml. Si el ML no está disponible, se usa la
    heurística de "la última mención cercana al final gana".
    """
    try:
        # Importación diferida para no forzar carga de DistilBETO
        # cuando no se necesita
        from src.verificador_birads_ml import predecir_birads_con_ml

        # Para cada candidato, extraer un fragmento de contexto
        scores_ml = []
        for candidato in candidatos:
            inicio_fragmento = max(0, candidato.posicion_inicio - 100)
            fin_fragmento = min(len(texto_completo), candidato.posicion_fin + 100)
            fragmento = texto_completo[inicio_fragmento:fin_fragmento]

            # ML predice sobre el fragmento
            prediccion, confianza = predecir_birads_con_ml(fragmento)

            # Puntuar: coincidencia + confianza
            if prediccion == candidato.valor:
                score_ml = confianza  # alta si el ML confirma esta mención
            else:
                score_ml = 1.0 - confianza  # baja si el ML predice otra cosa

            scores_ml.append((candidato, score_ml))

        # El candidato con mayor score gana
        scores_ml.sort(key=lambda x: x[1], reverse=True)
        ganador, score_ganador = scores_ml[0]

        return ganador, f"ml_ranking:score={score_ganador:.3f}"

    except (ImportError, Exception) as e:
        # Fallback: si el ML no está disponible, tomar la última mención
        # (heurística simple pero efectiva)
        ganador = max(candidatos, key=lambda m: m.posicion_relativa)
        return ganador, f"fallback_ultima_mencion (ml_error:{type(e).__name__})"


# =============================================================================
# FUNCIÓN PÚBLICA — BUSCAR BI-RADS FINAL
# =============================================================================

def buscar_birads_final(
    full_report: str,
    usar_ml_si_ambiguo: bool = True,
) -> Dict[str, Any]:
    """Encuentra el BI-RADS final del informe usando búsqueda híbrida.

    Args:
        full_report: texto completo del informe mamográfico.
        usar_ml_si_ambiguo: si True, activa DistilBETO como árbitro
            cuando hay candidatos con score empatado. Si False, en caso
            de empate se usa la heurística "última mención gana".

    Returns:
        Dict estructurado con:
            {
                "birads_final": int | None,
                "confianza": "alta" | "media" | "baja" | "no_detectado",
                "metodo": "busqueda_hibrida",
                "fase_ganadora": str,
                "ml_intervino": bool,
                "menciones_totales": int,
                "menciones_descartadas_count": int,
                "mencion_seleccionada": dict | None,
                "todas_las_menciones": list[dict],
            }
    """
    # ========================================================================
    # FASE 1 — Buscar todas las menciones
    # ========================================================================
    menciones = buscar_todas_menciones_birads(full_report)

    if not menciones:
        return _resultado_sin_birads_valido(
            razon="sin_menciones_birads",
            full_report=full_report,
        )

    # ========================================================================
    # FASE 2 — Filtrado contextual
    # ========================================================================
    menciones = filtrar_menciones_historicas(menciones)

    candidatos_validos = [m for m in menciones if not m.descartada]

    if not candidatos_validos:
        # Todas las menciones fueron descartadas por el filtrado contextual.
        # Descartar TODAS es poco fiable: el filtro puede haber sobre-disparado
        # (p. ej. "anterior" o "vs." en la ventana sin que la mención sea de
        # verdad histórica/comparativa). Es más seguro NO declarar omisión aquí
        # —hay menciones BI-RADS en el texto— y replegarse: se re-habilitan todas
        # las menciones como candidatos y se rankean por posición, con confianza
        # reducida. La omisión real se reserva para cuando NO hay ninguna mención
        # (Fase 1 vacía).
        for m in menciones:
            m.descartada = False
            m.razon_descarte = "rehabilitada_fase2_sobredescarte"
        candidatos_validos = menciones
        confianza_fallback = "baja"
    else:
        confianza_fallback = None

    # ========================================================================
    # FASE 3 — Ponderación posicional
    # ========================================================================
    candidatos_validos = ponderar_por_posicion(candidatos_validos, len(full_report))

    # Ordenar por score posicional (mayor primero)
    candidatos_validos.sort(key=lambda m: m.score_posicional, reverse=True)

    # Detectar empate en el máximo
    max_score = candidatos_validos[0].score_posicional
    empatados = [m for m in candidatos_validos if m.score_posicional == max_score]

    # ========================================================================
    # CASO A — Un solo candidato con score máximo
    # ========================================================================
    if len(empatados) == 1:
        ganador = empatados[0]
        return _construir_resultado(
            ganador=ganador,
            confianza=confianza_fallback or "alta",
            fase_ganadora=(
                "ponderacion_posicional_repliegue" if confianza_fallback
                else "ponderacion_posicional"
            ),
            ml_intervino=False,
            todas_menciones=menciones,
        )

    # ========================================================================
    # CASO B — Empate: activar ML como árbitro
    # ========================================================================
    if usar_ml_si_ambiguo and len(empatados) >= 2:
        ganador, motivo = rankear_con_ml(empatados, full_report)
        return _construir_resultado(
            ganador=ganador,
            confianza="media",  # Menor confianza porque hubo ambigüedad
            fase_ganadora=f"ml_ranking:{motivo}",
            ml_intervino=True,
            todas_menciones=menciones,
        )

    # ========================================================================
    # CASO C — Empate sin ML (fallback): última mención gana
    # ========================================================================
    ganador = max(empatados, key=lambda m: m.posicion_relativa)
    return _construir_resultado(
        ganador=ganador,
        confianza="media",
        fase_ganadora="fallback_ultima_mencion_de_empate",
        ml_intervino=False,
        todas_menciones=menciones,
    )


# =============================================================================
# FUNCIONES AUXILIARES INTERNAS
# =============================================================================

def _construir_resultado(
    ganador: MencionBirads,
    confianza: str,
    fase_ganadora: str,
    ml_intervino: bool,
    todas_menciones: List[MencionBirads],
) -> Dict[str, Any]:
    """Empaqueta el resultado de la búsqueda híbrida."""
    return {
        "birads_final": ganador.valor,
        "confianza": confianza,
        "metodo": "busqueda_hibrida",
        "fase_ganadora": fase_ganadora,
        "ml_intervino": ml_intervino,
        "menciones_totales": len(todas_menciones),
        "menciones_descartadas_count": sum(1 for m in todas_menciones if m.descartada),
        "mencion_seleccionada": ganador.to_dict(),
        "todas_las_menciones": [m.to_dict() for m in todas_menciones],
    }


def _construir_resultado_vacio(
    razon: str,
    todas_menciones: Optional[List[MencionBirads]] = None,
) -> Dict[str, Any]:
    """Empaqueta un resultado cuando no se encontró BI-RADS válido."""
    todas_menciones = todas_menciones or []
    return {
        "birads_final": None,
        "confianza": "no_detectado",
        "metodo": "busqueda_hibrida",
        "fase_ganadora": None,
        "ml_intervino": False,
        "menciones_totales": len(todas_menciones),
        "menciones_descartadas_count": sum(1 for m in todas_menciones if m.descartada),
        "razon_no_detectado": razon,
        "mencion_seleccionada": None,
        "todas_las_menciones": [m.to_dict() for m in todas_menciones],
    }


def _resultado_sin_birads_valido(
    razon: str,
    full_report: str,
    todas_menciones: Optional[List[MencionBirads]] = None,
) -> Dict[str, Any]:
    """FASE 4 (parte B): decide entre OMISIÓN (con alerta) o vacío trivial.

    Cuando no hay un BI-RADS válido, evalúa si el informe tiene hallazgos o
    acciones clínicas. Si los tiene, es una omisión y se agrega la clave
    'alerta' (que el verificador propaga como alerta_omision). Si no, se
    devuelve el resultado vacío ordinario (informe no procesable).
    """
    resultado = _construir_resultado_vacio(razon=razon, todas_menciones=todas_menciones)

    evidencia = detectar_hallazgos_reportables(full_report)
    if evidencia is None:
        return resultado

    resultado["alerta"] = {
        "tipo": "omision_birads",
        "mensaje": (
            "El informe describe hallazgos radiológicos y/o recomendaciones "
            "clínicas pero no tiene una categoría BI-RADS asignada. "
            "Posible omisión que requiere revisión del radiólogo antes de la entrega."
        ),
        "severidad": evidencia["severidad"],
        "accion_sugerida": (
            "Revisar el informe y asignar la categoría BI-RADS correspondiente "
            "antes de finalizar/entregar el estudio."
        ),
        "evidencia": {
            "hallazgos_detectados": evidencia["hallazgos_detectados"],
            "acciones_detectadas": evidencia["acciones_detectadas"],
            "razon_buscador": razon,
        },
    }
    return resultado


# =============================================================================
# CLI (opcional)
# =============================================================================

def _ejecutar_cli(argv: Optional[list] = None) -> int:
    """Punto de entrada del CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.buscador_birads",
        description="Búsqueda híbrida de BI-RADS final en informes mamográficos.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Ruta a un archivo TXT con el informe (opcional).",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Desactivar ML en caso de empate.",
    )

    args = parser.parse_args(argv)

    if args.input is None:
        # Sin argumentos: ejecutar tests inline
        _ejecutar_tests()
        return 0

    # Leer el archivo
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            full_report = f.read()
    except Exception as e:
        print(f"Error al leer archivo: {e}", file=sys.stderr)
        return 1

    resultado = buscar_birads_final(
        full_report=full_report,
        usar_ml_si_ambiguo=not args.no_ml,
    )

    print(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))
    return 0


# =============================================================================
# TESTS INLINE
# =============================================================================

INFORME_ESTANDAR = """
INFORME DE MAMOGRAFIA

HALLAZGOS: Mama densa heterogenea. Sin nodulos sospechosos.

CONCLUSION: BI-RADS 2 - Hallazgos benignos.
"""

INFORME_SIN_ENCABEZADO_CONCLUSION = """
Paciente de 52 años. Mamografia bilateral.

Mama derecha: densidad heterogenea, sin lesiones focales.
Mama izquierda: microcalcificaciones agrupadas en cuadrante superior externo.

Se sugiere biopsia estereotactica.

BI-RADS 4
"""

INFORME_CON_COMPARACION_HISTORICA = """
INFORME DE MAMOGRAFIA

Comparado con estudio anterior BI-RADS 2 de junio 2024, se observa
aparicion de microcalcificaciones agrupadas.

CONCLUSION: BI-RADS 4 - Hallazgo sospechoso, requiere estudio histologico.
"""

INFORME_MULTIPLES_MENCIONES = """
INFORME DE MAMOGRAFIA

HALLAZGOS: Densidad asimetrica en cuadrante superior externo derecho.
Segun el sistema BI-RADS 0 indica estudio incompleto.

Estudio previo BI-RADS 2 hace 12 meses.

CONCLUSION: BI-RADS 0 - Estudio incompleto, requiere ecografia dirigida.
"""

INFORME_SIN_BIRADS = """
INFORME DE MAMOGRAFIA

Mama densa heterogenea. Sin nodulos sospechosos.

Se sugiere control anual.
"""

INFORME_CON_NEGACION = """
INFORME DE MAMOGRAFIA

Nodulo ovalado circunscrito de aspecto benigno.

CONCLUSION: corresponde a BI-RADS 2. Los hallazgos no son sugerentes de BI-RADS 4.
"""

INFORME_NEGACION_FALSO_POSITIVO = """
INFORME DE MAMOGRAFIA

No se observa nodulo sospechoso ni masa. Se sugiere biopsia para descartar patologia.

CONCLUSION: BI-RADS 4 - hallazgo sospechoso.
"""


def _ejecutar_tests() -> None:
    """Suite de tests inline. Ejecutar con: python -m src.buscador_birads"""
    print("=" * 75)
    print("TESTS DE src/buscador_birads.py")
    print("=" * 75)

    casos = [
        {
            "nombre": "T1: Informe estandar con encabezado CONCLUSION",
            "input": INFORME_ESTANDAR,
            "birads_esperado": 2,
        },
        {
            "nombre": "T0: BI-RADS en numeros ROMANOS (BIRADS II ACR b)",
            "input": ("Conclusion: Examen sin hallazgos sospechosos de lesion "
                      "maligna. BIRADS II ACR b"),
            "birads_esperado": 2,
        },
        {
            "nombre": "T0b: BI-RADS con modalidad (BI-RADS US 2)",
            "input": ("Impresion: Examen sin hallazgos sugerentes de "
                      "malignidad. BI-RADS US 2"),
            "birads_esperado": 2,
        },
        {
            "nombre": "T0c: BI-RADS modalidad con guion + romano (Birads -us II)",
            "input": "Impresion diagnostica: Examen sin hallazgos. Birads -us II",
            "birads_esperado": 2,
        },
        {
            "nombre": "T2: Sin encabezado CONCLUSION, BI-RADS al final",
            "input": INFORME_SIN_ENCABEZADO_CONCLUSION,
            "birads_esperado": 4,
        },
        {
            "nombre": "T3: Comparacion historica + conclusion",
            "input": INFORME_CON_COMPARACION_HISTORICA,
            "birads_esperado": 4,  # NO debe elegir el 2 histórico
        },
        {
            "nombre": "T4: Multiples menciones (educacional + historica + real)",
            "input": INFORME_MULTIPLES_MENCIONES,
            "birads_esperado": 0,  # Debe elegir el de la conclusion final
        },
        {
            "nombre": "T5: Sin menciones BI-RADS",
            "input": INFORME_SIN_BIRADS,
            "birads_esperado": None,
        },
        {
            "nombre": "T6: Negacion (descarta BI-RADS 4 negado, elige el 2)",
            "input": INFORME_CON_NEGACION,
            "birads_esperado": 2,  # el "no son sugerentes de BI-RADS 4" se descarta
        },
        {
            "nombre": "T7: Negacion sin falso positivo ('no se observa', 'descartar')",
            "input": INFORME_NEGACION_FALSO_POSITIVO,
            "birads_esperado": 4,  # NO debe descartar el 4 por 'descartar patologia'
        },
    ]

    n_pasados = 0
    for caso in casos:
        try:
            resultado = buscar_birads_final(
                full_report=caso["input"],
                usar_ml_si_ambiguo=False,  # Sin ML para tests deterministas
            )
        except Exception as e:
            print(f"  [FALLA] {caso['nombre']}: excepción {e}")
            continue

        birads_obtenido = resultado["birads_final"]
        esperado = caso["birads_esperado"]

        if birads_obtenido == esperado:
            n_pasados += 1
            print(f"  [PASA] {caso['nombre']}")
            print(f"         BI-RADS={birads_obtenido}, "
                  f"fase={resultado.get('fase_ganadora')}, "
                  f"total={resultado['menciones_totales']}, "
                  f"descartadas={resultado['menciones_descartadas_count']}")
        else:
            print(f"  [FALLA] {caso['nombre']}")
            print(f"          Esperado: {esperado}, obtenido: {birads_obtenido}")
            print(f"          Fase: {resultado.get('fase_ganadora')}")
            if resultado.get("todas_las_menciones"):
                for m in resultado["todas_las_menciones"]:
                    marca = "[descartada]" if m["descartada"] else "[valida]"
                    print(f"            {marca} BI-RADS {m['valor']} "
                          f"pos={m['posicion_relativa']:.2f} "
                          f"score={m['score_posicional']}")

    print(f"\nResumen: {n_pasados}/{len(casos)} tests pasados")

    if n_pasados == len(casos):
        print("Estado: OK — buscador_birads.py listo para integracion.")
    else:
        print("Estado: FALLA — revisar los casos que no pasaron.")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    sys.exit(_ejecutar_cli())
