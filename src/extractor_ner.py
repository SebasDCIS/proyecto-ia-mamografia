"""
Extractor de recomendaciones por NER (DistilBETO fine-tuneado).

Actúa como VÍA DE RESPALDO de la extracción por reglas: cuando la regex no
logra localizar la recomendación con confianza, este módulo intenta extraerla
con el modelo entrenado (notebook 11), que generaliza a redacciones no vistas.

Diseño defensivo:
  - Carga perezosa (lazy): el modelo solo se carga la primera vez que se usa.
  - Respaldo elegante: si el modelo no está disponible (archivos ausentes, sin
    torch/transformers, etc.), las funciones devuelven "no encontrado" sin
    lanzar excepción, de modo que el pipeline sigue funcionando con reglas.
  - Sin efectos colaterales en import: importar este módulo NO carga el modelo
    ni requiere torch. Eso mantiene el resto del sistema y los tests intactos.
"""

from typing import Any, Dict, Optional
import os

# Ruta por defecto del modelo entrenado (relativa a la raíz del proyecto).
RUTA_MODELO_NER = os.environ.get(
    "NER_MODELO_PATH", "models/ner_recomendacion_final"
)

# Estado global de carga (se inicializa perezosamente).
_MODELO = None
_TOKENIZER = None
_ID2LABEL = None
_DISPONIBLE: Optional[bool] = None  # None = aún no probado
_MAX_LEN = 384


def ner_disponible(ruta_modelo: str = RUTA_MODELO_NER) -> bool:
    """Indica si el modelo NER puede usarse (intenta cargarlo una sola vez)."""
    return _cargar_modelo(ruta_modelo)


def _cargar_modelo(ruta_modelo: str = RUTA_MODELO_NER) -> bool:
    """Carga perezosa del modelo. Devuelve True si quedó disponible."""
    global _MODELO, _TOKENIZER, _ID2LABEL, _DISPONIBLE
    if _DISPONIBLE is not None:
        return _DISPONIBLE
    try:
        if not os.path.isdir(ruta_modelo):
            _DISPONIBLE = False
            return False
        # Imports dentro de la función: no se exige torch salvo que se use el NER.
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
        )
        _TOKENIZER = AutoTokenizer.from_pretrained(ruta_modelo)
        _MODELO = AutoModelForTokenClassification.from_pretrained(ruta_modelo)
        _MODELO.eval()
        _ID2LABEL = _MODELO.config.id2label
        _DISPONIBLE = True
    except Exception:
        # Cualquier fallo (sin torch, modelo corrupto, etc.) -> respaldo elegante.
        _DISPONIBLE = False
    return _DISPONIBLE


def extraer_recomendacion_ner(
    texto_informe: str,
    ruta_modelo: str = RUTA_MODELO_NER,
) -> Dict[str, Any]:
    """Extrae el span de recomendación de un informe usando el modelo NER.

    Returns:
        Dict con:
            - texto (str): span extraído (vacío si no encontró o no disponible)
            - encontrado (bool)
            - fuente (str): 'ner_distilbeto' | 'ner_no_disponible' | 'ner_vacio'
            - confianza (str): 'alta' | 'no_aplica'
    """
    base = {"texto": "", "encontrado": False,
            "fuente": "ner_no_disponible", "confianza": "no_aplica"}

    if not isinstance(texto_informe, str) or not texto_informe.strip():
        return base
    if not _cargar_modelo(ruta_modelo):
        return base

    try:
        import torch
        palabras = texto_informe.split()
        tok = _TOKENIZER(
            palabras, truncation=True, max_length=_MAX_LEN,
            is_split_into_words=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = _MODELO(**tok).logits
        preds = logits.argmax(-1)[0].tolist()
        word_ids = tok.word_ids()

        # Etiqueta por palabra (primer subtoken de cada palabra)
        etiqueta_palabra: Dict[int, str] = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in etiqueta_palabra:
                etiqueta_palabra[wid] = _ID2LABEL[preds[idx]]

        # Tomar SOLO el primer bloque CONTIGUO de recomendación (B-REC..I-REC),
        # no unir fragmentos dispersos. Esto evita arrastrar firmas del médico o
        # texto de otro informe si el modelo etiqueta palabras sueltas por error.
        palabras_ord = sorted(etiqueta_palabra)
        inicio = None
        fin = None
        for w in palabras_ord:
            et = etiqueta_palabra[w]
            if et != "O" and inicio is None:
                inicio = w   # comienza el span
                fin = w
            elif et != "O" and inicio is not None:
                # solo extiende si es CONTIGUA a la palabra anterior del span
                if w == fin + 1:
                    fin = w
                else:
                    break    # hueco -> termina el primer bloque contiguo
            elif et == "O" and inicio is not None:
                break        # primera O tras iniciar -> fin del bloque

        if inicio is None:
            return {"texto": "", "encontrado": False,
                    "fuente": "ner_vacio", "confianza": "no_aplica"}

        span = [palabras[w] for w in range(inicio, fin + 1)]
        texto_span = " ".join(span).strip()

        if not texto_span:
            return {"texto": "", "encontrado": False,
                    "fuente": "ner_vacio", "confianza": "no_aplica"}
        return {"texto": texto_span, "encontrado": True,
                "fuente": "ner_distilbeto", "confianza": "alta"}
    except Exception:
        return base
