#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablación por enmascaramiento — ¿DistilBETO sufre fuga de etiqueta?

Prueba si el modelo aprendió a EVALUAR los hallazgos o solo a COPIAR el número
BI-RADS que ya está escrito en el texto de entrada.

Procedimiento:
  1. Reconstruye el test set del notebook 04 (mismo seed, mismo split).
  2. Evalúa el modelo sobre el test ORIGINAL (con la mención BI-RADS visible).
  3. Enmascara la mención BI-RADS en cada informe del test.
  4. Evalúa el modelo sobre el test ENMASCARADO.
  5. Compara los Macro F1.

Interpretación:
  - Si el F1 se MANTIENE alto sin el número  -> el modelo predice desde los
    hallazgos. NO hay fuga de etiqueta. Defendible y ahora DEMOSTRADO.
  - Si el F1 se DESPLOMA hacia el azar        -> el modelo copiaba el número.
    Hay fuga. Toca reentrenar enmascarando la entrada.

Ejecutar desde la raíz del repo (con el .venv activo):
    python ablacion_leakage_birads.py
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — igual que en el notebook 04
# ---------------------------------------------------------------------------
SEED = 42
DATA_PATH = "data/processed/reports_cleaned.csv"   # ajusta si tu ruta difiere
CHECKPOINT = "notebooks/results_distilbeto/best_model"  # o checkpoint-1743
MAX_LENGTH = 256

# ---------------------------------------------------------------------------
# ENMASCARAMIENTO DE LA MENCIÓN BI-RADS
# ---------------------------------------------------------------------------
# Reemplaza cualquier forma de "BI-RADS X" por un marcador neutro que conserva
# la estructura de la frase pero elimina el NÚMERO (la respuesta).
PATRONES_BIRADS = [
    re.compile(r"bi\s*[-*.]?\s*rad[ais]?[s]?\s*®?\s*:?\s*\(?\s*"
               r"([0-6]|VI|IV|V|III|II|I|cero|uno|dos|tres|cuatro|cinco|seis)"
               r"(\s*-\s*[0-6])?"      # rango opcional (p. ej. 4-5)
               r"\s*([abc])?\s*\)?", re.IGNORECASE),
    re.compile(r"categor[íi]a\s*[-:]?\s*([0-6])(\s*-\s*[0-6])?\b", re.IGNORECASE),
    re.compile(r"\bACR\s*[-:]?\s*([0-6])(\s*-\s*[0-6])?\b", re.IGNORECASE),
]

MARCADOR = "categoria oculta"


def enmascarar_birads(texto: str) -> str:
    """Elimina el número BI-RADS del texto, dejando un marcador neutro."""
    t = texto
    for patron in PATRONES_BIRADS:
        t = patron.sub(MARCADOR, t)
    return t


# ---------------------------------------------------------------------------
# PREDICCIÓN EN LOTE
# ---------------------------------------------------------------------------
def predecir(textos, tokenizer, model, device, batch_size=16):
    preds = []
    model.eval()
    for i in range(0, len(textos), batch_size):
        lote = list(textos[i:i + batch_size])
        inputs = tokenizer(lote, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
    return np.array(preds)


def main():
    print("=" * 68)
    print("ABLACIÓN POR ENMASCARAMIENTO — ¿fuga de etiqueta en DistilBETO?")
    print("=" * 68)

    # 1) Reconstruir el test set (mismo seed que el notebook 04)
    df = pd.read_csv(DATA_PATH)
    X = df["Full_Report_clean"].astype(str).values
    y = df["BI-RADS"].astype(int).values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\nTest set reconstruido: {len(X_test)} ejemplos")

    # Diagnóstico rápido: ¿el texto contiene la mención BI-RADS?
    frac_con_birads = np.mean([bool(re.search(r"rad", t, re.IGNORECASE)) for t in X_test])
    print(f"Fracción del test que menciona 'rad(s)' en el texto: {frac_con_birads:.1%}")
    # Cuántos textos cambian al enmascarar (mención efectivamente presente)
    X_test_masked = np.array([enmascarar_birads(t) for t in X_test])
    n_cambiados = int(np.sum([a != b for a, b in zip(X_test, X_test_masked)]))
    print(f"Informes donde el enmascarado modificó el texto: {n_cambiados}/{len(X_test)} "
          f"({100*n_cambiados/len(X_test):.1f}%)")

    # 2) Cargar modelo
    print(f"\nCargando modelo desde: {CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    print(f"Device: {device}")

    # 3) Evaluar SIN enmascarar (original)
    preds_orig = predecir(X_test, tokenizer, model, device)
    f1_orig = f1_score(y_test, preds_orig, average="macro", zero_division=0)
    acc_orig = accuracy_score(y_test, preds_orig)

    # 4) Evaluar CON enmascarado
    preds_mask = predecir(X_test_masked, tokenizer, model, device)
    f1_mask = f1_score(y_test, preds_mask, average="macro", zero_division=0)
    acc_mask = accuracy_score(y_test, preds_mask)

    # 5) Comparar
    print("\n" + "=" * 68)
    print("RESULTADOS")
    print("=" * 68)
    print(f"  {'':<22}{'Macro F1':>12}{'Accuracy':>12}")
    print(f"  {'Texto ORIGINAL':<22}{f1_orig:>12.4f}{acc_orig:>12.4f}")
    print(f"  {'Texto ENMASCARADO':<22}{f1_mask:>12.4f}{acc_mask:>12.4f}")
    caida = f1_orig - f1_mask
    print(f"  {'Caída de Macro F1':<22}{caida:>12.4f}  ({100*caida/max(f1_orig,1e-9):.1f}%)")

    # Interpretación automática
    print("\n" + "-" * 68)
    print("INTERPRETACIÓN")
    print("-" * 68)
    if caida < 0.05:
        print("  El rendimiento se MANTIENE sin el número visible.")
        print("  -> El modelo predice desde los hallazgos, NO copia la etiqueta.")
        print("  -> NO hay fuga de etiqueta relevante. RESULTADO DEFENDIBLE.")
    elif caida < 0.20:
        print("  Caída MODERADA. El modelo usa parcialmente el número, pero también")
        print("  información de los hallazgos. Fuga parcial: conviene reentrenar")
        print("  enmascarando la entrada y volver a medir.")
    else:
        print("  Caída GRANDE. El modelo dependía fuertemente del número escrito.")
        print("  -> Hay FUGA DE ETIQUETA. Reentrenar enmascarando la mención BI-RADS")
        print("     en la entrada (o entrenando solo con la sección de hallazgos).")

    print("\nDetalle por clase (texto enmascarado):")
    print(classification_report(y_test, preds_mask, zero_division=0))


if __name__ == "__main__":
    main()
