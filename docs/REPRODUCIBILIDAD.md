# Protocolo de reproducibilidad

Este documento reúne lo necesario para repetir cada cifra del informe: entorno,
datos, orden de ejecución, parámetros y resultado esperado. Complementa al
[`README.md`](../README.md) y a la [bitácora](BITACORA.md).

Todas las particiones y entrenamientos usan la semilla **42**.

---

## 1. Verificación mínima (sin datos ni GPU)

Solo requiere `numpy` y `scikit-learn`. Verifica el pipeline reglado completo y la
cobertura de formatos, que son las vías que operan en producción.

```bash
python -m src.predict                    # esperado: 8/8 tests pasados
python -m tests.casos_formato_chileno    # esperado: 17/17 en las 4 dimensiones
```

Los 17 casos son sintéticos (nombres, RUT y fechas inventados).

## 2. Entorno

| Elemento | Valor |
|---|---|
| Python | 3.9 |
| Dependencias | versiones exactas en [`requirements.txt`](../requirements.txt) |
| Librerías principales | scikit-learn 1.6.1, PyTorch 2.8.0, Transformers 4.57.6, pandas 2.3.3, numpy 2.0.2 |
| Hardware principal | Mac con Apple Silicon, aceleración MPS |
| Hardware secundario | GPU en Google Colab (notebooks `*_Colab`) |
| Modelo base | `dccuchile/distilbert-base-spanish-uncased` (DistilBETO), se descarga de Hugging Face |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Las operaciones con GPU o MPS no son del todo deterministas, por lo que al
repetir un entrenamiento los valores pueden variar levemente en los últimos
decimales. Las cifras de las reglas (M1, M2 por reglas, M3) no dependen del
hardware y se reproducen exactas.

## 3. Datos

El corpus no se incluye en el repositorio.

1. Descargar de Zenodo: [10.5281/zenodo.14827680](https://doi.org/10.5281/zenodo.14827680)
   (artículo: [10.1016/j.dib.2025.111761](https://doi.org/10.1016/j.dib.2025.111761)).
2. Generar `data/processed/reports_cleaned.csv` ejecutando
   `notebooks/00_exploracion_informes.ipynb` y `notebooks/01_limpieza.ipynb`.
3. El archivo debe tener, entre otras, las columnas `Full_Report_clean`,
   `Recommendations_clean` y `BI-RADS`.

**Comprobaciones de que el archivo es el correcto:**

| Comprobación | Valor esperado |
|---|---|
| Filas | 4 357 |
| Filas con recomendación | 4 347 |
| Distribución por BI-RADS | 0 = 966 · 1 = 596 · 2 = 2 635 · 3 = 87 · 4 = 52 · 5 = 16 · 6 = 5 |

## 4. Recuento de las cifras del pipeline reglado

```bash
python scripts/recuento_cifras.py
```

Salida esperada sobre el corpus completo (4 357 informes, 0 errores):

| Cifra | Valor |
|---|---|
| Confianza de M1 | alta 4 355 (99,95 %) · media 2 |
| Recomendación clasificada | 4 347 de 4 357 (99,77 %) |
| Estados del cotejo | coherente 2 800 · coherente_equivalente 1 455 · coherente_con_precaucion 44 · incoherente 50 · revision_extraccion 8 |
| Incoherentes por severidad | crítica 19 · alta 18 · media 9 · baja 4 (50 = 1,15 %) |
| Revisión por extracción | 8 (0,18 %) |

## 5. Experimentos, en el orden en que se ejecutaron

| Notebook | Tarea | Partición | Parámetros | Resultado esperado |
|---|---|---|---|---|
| `02_baseline_tfidf` | Línea base clásica | ver notebook | LinearSVC con `class_weight='balanced'` | Macro F1 0,9367 (CV 5-fold: 0,7369 ± 0,0703) |
| `03_transformers` | DistilBERT en inglés | 80/20 estratificada, semilla 42 | `distilbert-base-uncased` | Macro F1 0,471 (exactitud 0,932) |
| `04_distilbeto` | DistilBETO en español | 80/20 estratificada, semilla 42 | 3 épocas, lote 8, tasa de aprendizaje 2×10⁻⁵, 256 subtokens, aumentación por sinónimos clínicos | Macro F1 0,9386 (step 1 743, 872 informes de prueba) |
| `04b_cv_distilbeto` | Validación cruzada | `StratifiedKFold` 5, `shuffle=True`, semilla 42 | Igual que `04`; aumentación **dentro** de cada fold | Macro F1 0,8877 ± 0,0501 |
| `04c_cv_ventana_local` | CV sobre la ventana real del verificador | 5-fold, deduplicado por texto exacto | Ventana de 250 caracteres (−200/+50) | Ventana 0,9958 ± 0,0043 · informe completo 0,8920 ± 0,0805 |
| `05_extractor_birads` | Extractor por reglas | corpus completo | sin entrenamiento | Macro F1 0,9995 |
| `07_validacion_cotejo_acr` | Cotejo end-to-end | corpus completo | tabla ACR | validación del MVP |
| `08_verificador_birads_ml` | Verificador (retirado) | — | — | se conserva para reproducir su evaluación |
| `11_extractor_ner_recomendacion` | NER de recomendación | estratificada por BI-RADS, 70/15/15, semilla 42, **tras deduplicar** | tasa de aprendizaje 3×10⁻⁵, lote 16, hasta 6 épocas, decaimiento de pesos 0,01, calentamiento 10 %, parada temprana con paciencia 2 sobre F1 de validación, 384 subtokens | F1 de span 0,9991 (580 de 4 345 duplicados eliminados) |
| `11b_ablacion_ner` | Ablación del encabezado | test del NER | sin reentrenar | 1,0000 → 1,0000 |
| `11c_estres_ner` | Prueba de estrés | 565 informes de prueba | quitar encabezado y/o cambiar el verbo | control 1,0000/1,0000 · sin encabezado 0,9460/1,0000 · verbo nuevo 1,0000/0,9929 · ambos 0,5314/0,9956 (regla/NER) |
| `Matching_Embeddings_Clinico_Colab` | Embeddings como clasificador | corpus completo | 38 anclas, 8 categorías, umbral fijado en 90 % | 77,4 % (descartado) |
| `Segundo_Revisor_Embeddings_Colab` | Embeddings como segundo revisor | corpus completo | umbral fijado en 10 % | 26,97 % de discrepancias (descartado) |
| `Entrenamiento_BIRADS_Colab` | Predicción desde hallazgos | CV 5-fold estratificada | focal loss, pesos de clase, calibración por temperatura (T = 0,595) | Macro F1 fuera de partición 0,6236 |

## 6. Ablación del verificador

```bash
python scripts/ablacion_leakage_birads.py
```

Requiere el checkpoint entrenado en `notebooks/results_distilbeto/best_model`
(se obtiene con `04_distilbeto`). Esperado: Macro F1 0,9386 con el número visible
y 0,5438 con el número enmascarado (871 de 872 informes tenían la mención).

## 7. Modelos entrenados

Los pesos no se versionan (`models/` está en `.gitignore`, pesan cerca de 250 MB).
El NER se regenera ejecutando `11_extractor_ner_recomendacion`, que deja el modelo
en `models/ner_recomendacion_final`. Si no existe, el pipeline funciona solo con
reglas y no se interrumpe.

## 8. Dónde nace cada cifra del informe

| Cifra | Origen |
|---|---|
| Macro F1 de extracción 0,9995 | `05_extractor_birads` |
| 0,939 → 0,544 | `scripts/ablacion_leakage_birads.py` |
| 99,82 % frente a 99,78 % | ventana de 250 caracteres, `04c_cv_ventana_local` y bitácora |
| 0,9386 → 0,8877 | `04_distilbeto` y `04b_cv_distilbeto` |
| NER 0,9991 y estrés 0,5314 / 0,9956 | `11`, `11b` y `11c` |
| 50 alertas (1,15 %), 19 críticas, 8 revisiones | `scripts/recuento_cifras.py` |
| 0,624 (predicción) | `Entrenamiento_BIRADS_Colab` |
| 17/17 | `tests/casos_formato_chileno.py` |

## 9. Lo que este protocolo no cubre

- No hay corpus chileno anotado: la validación en el contexto de despliegue sigue
  pendiente. La batería de 17 casos verifica cobertura de formatos, no desempeño
  poblacional.
- La prueba de estrés del NER es una simulación sobre el corpus paraguayo
  perturbado.
- Las etiquetas del NER salen del campo de recomendación del propio corpus, no de
  una anotación independiente.
