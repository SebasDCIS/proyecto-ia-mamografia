#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnóstico del modelo ML (apoyo de lectura del BI-RADS).

Responde dos preguntas:
  1. ¿El mapeo de etiquetas del modelo es correcto? (posible bug de índices)
  2. ¿El modelo concuerda sobre informes REALES del dataset (in-distribution)?

Uso (desde la raíz del proyecto, con el .venv activo):
    python diagnostico_ml.py                      # busca reports_cleaned.csv solo
    python diagnostico_ml.py /ruta/al/reports_cleaned.csv
    python diagnostico_ml.py /ruta/al/csv 50      # con N informes (default 30)
"""

import os
import sys
import glob

import pandas as pd

# --- Localizar el CSV ---
def encontrar_csv(arg):
    if arg and os.path.exists(arg):
        return arg
    candidatos = [
        "reports_cleaned.csv",
        "data/reports_cleaned.csv",
        os.path.expanduser("~/Downloads/reports_cleaned.csv"),
        os.path.expanduser("~/Documents/Proyectos_Doc/proyecto-ia-mamografia/reports_cleaned.csv"),
    ]
    candidatos += glob.glob("**/reports_cleaned.csv", recursive=True)
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None


ruta_csv = encontrar_csv(sys.argv[1] if len(sys.argv) > 1 else None)
n_muestra = int(sys.argv[2]) if len(sys.argv) > 2 else 30

if ruta_csv is None:
    print("ERROR: no encontré reports_cleaned.csv. Pásalo como argumento:")
    print("       python diagnostico_ml.py /ruta/a/reports_cleaned.csv")
    sys.exit(1)

print("=" * 68)
print("DIAGNÓSTICO DEL MODELO ML — APOYO DE LECTURA DEL BI-RADS")
print("=" * 68)
print(f"CSV: {ruta_csv}\n")

# --- 1) Inspeccionar el mapeo de etiquetas del modelo ---
print("[1] MAPEO DE ETIQUETAS DEL MODELO")
print("-" * 68)
try:
    from src.verificador_birads_ml import _cargar_modelo, _MODELO_CARGADO, _predecir_birads_ml
    _cargar_modelo()
    modelo = _MODELO_CARGADO["modelo"]
    cfg = modelo.config
    id2label = dict(cfg.id2label)
    print(f"num_labels : {cfg.num_labels}")
    print(f"id2label   : {id2label}")

    # ¿Es un mapeo identidad (índice == BI-RADS)?
    def val_label(v):
        s = str(v)
        return "".join(ch for ch in s if ch.isdigit())
    identidad = all(str(i) == val_label(lab) for i, lab in id2label.items()) \
        or all(lab in (f"LABEL_{i}",) for i, lab in id2label.items())
    if identidad:
        print("\n=> Mapeo IDENTIDAD (o LABEL_i en orden): el argmax = BI-RADS. OK.")
    else:
        print("\n=> ¡ATENCIÓN! El mapeo NO es identidad.")
        print("   El código hace int(argmax) directo, así que estaría prediciendo")
        print("   la categoría EQUIVOCADA. Este es el bug: hay que mapear el argmax")
        print("   a través de id2label.")
except Exception as e:
    print(f"No se pudo inspeccionar el modelo: {e}")
    sys.exit(1)

# --- 2) Concordancia sobre informes reales (in-distribution) ---
print("\n[2] CONCORDANCIA SOBRE INFORMES REALES (in-distribution)")
print("-" * 68)
df = pd.read_csv(ruta_csv)
col_texto = "Full_Report_clean" if "Full_Report_clean" in df.columns else "Full_Report"
muestra = df.sample(min(n_muestra, len(df)), random_state=1)

ok = 0
por_clase = {}
errores_ejemplo = []
for _, r in muestra.iterrows():
    real = None
    try:
        real = int(str(r["BI-RADS"]).strip()[0])
    except Exception:
        continue
    pred = _predecir_birads_ml(str(r[col_texto]))["birads_predicho"]
    por_clase.setdefault(real, [0, 0])
    por_clase[real][1] += 1
    if pred == real:
        ok += 1
        por_clase[real][0] += 1
    elif len(errores_ejemplo) < 5:
        errores_ejemplo.append((real, pred, str(r[col_texto])[:90]))

total = sum(v[1] for v in por_clase.values())
print(f"Concordancia global: {ok}/{total}  ({100*ok/max(total,1):.0f}%)\n")
print("Por clase real (aciertos/total):")
for k in sorted(por_clase):
    a, t = por_clase[k]
    print(f"  BI-RADS {k}: {a}/{t}")

if errores_ejemplo:
    print("\nEjemplos donde difirió (real -> predicho):")
    for real, pred, txt in errores_ejemplo:
        print(f"  {real} -> {pred} | {txt}...")

# --- Veredicto ---
print("\n" + "=" * 68)
print("VEREDICTO")
print("-" * 68)
pct = 100 * ok / max(total, 1)
if not identidad:
    print("Hay un BUG de mapeo de etiquetas (ver sección [1]). Arreglar eso primero.")
elif pct >= 80:
    print(f"El modelo concuerda bien ({pct:.0f}%) sobre informes reales.")
    print("=> NO hay bug. Lo que veías era tus casos sintéticos fuera de distribución.")
    print("   Camino: semáforo de confianza + demostrar con informes reales.")
else:
    print(f"Concordancia baja ({pct:.0f}%) incluso sobre informes reales.")
    print("=> Revisar: mapeo de etiquetas, max_length (256 vs 512), o el preprocesamiento.")
print("=" * 68)
