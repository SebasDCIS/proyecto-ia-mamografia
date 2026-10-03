#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recuenta las cifras que se citan en README, informe y bitácora, con el pipeline
actual (solo reglas) sobre el corpus completo.

Ejecutar desde la raíz del repositorio, con el .venv activo:
    python scripts/recuento_cifras.py
    python scripts/recuento_cifras.py --csv data/processed/reports_cleaned.csv
"""
import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import procesar_informe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/reports_cleaned.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    conf = collections.Counter()
    estados = collections.Counter()
    sev_incoh = collections.Counter()
    con_rec = 0
    rec_clasificada = 0
    errores = 0

    for i, fila in df.iterrows():
        rec = fila.get("Recommendations_clean")
        rec = None if pd.isna(rec) else str(rec)
        try:
            r = procesar_informe(
                str(fila["Full_Report_clean"]),
                recommendations_col=rec,
                informe_id=str(i),
                usar_verificador_ml=False,
                usar_ner_recomendacion=False,
            )
        except Exception:
            errores += 1
            continue
        conf[(r.get("birads") or {}).get("confianza")] += 1
        if rec:
            con_rec += 1
            if (r.get("recomendacion") or {}).get("categoria_principal"):
                rec_clasificada += 1
        c = r.get("cotejo_acr") or {}
        estados[c.get("estado")] += 1
        if c.get("estado") == "incoherente":
            sev_incoh[c.get("severidad")] += 1
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{n}", flush=True)

    pct = lambda a, b: f"{100 * a / b:.2f} %"
    print("\n" + "=" * 60)
    print(f"Informes procesados: {n}   (errores: {errores})")
    print("-" * 60)
    print("Confianza de la extracción BI-RADS:", dict(conf))
    alta = conf.get("alta", 0)
    print(f"  confianza alta: {alta}  ({pct(alta, n)})")
    print("-" * 60)
    print(f"Con recomendación en el corpus: {con_rec}")
    print(f"  recomendación clasificada: {rec_clasificada} de {n}  ({pct(rec_clasificada, n)})")
    print("-" * 60)
    print("Estados del cotejo:", dict(estados))
    inc = estados.get("incoherente", 0)
    print(f"  incoherentes: {inc}  ({pct(inc, n)})   por severidad: {dict(sev_incoh)}")
    rev = estados.get("revision_extraccion", 0)
    print(f"  revisión por extracción: {rev}  ({pct(rev, n)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
