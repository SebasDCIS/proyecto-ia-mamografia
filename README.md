# Proyecto IA Mamografía

Sistema local de estructuración y validación técnica de informes mamográficos en español mediante Procesamiento de Lenguaje Natural.

**Curso:** BME513 — Inteligencia Artificial en Salud
**Programa:** Doctorado en Ciencia, Innovación y Salud (DCIS) — Universidad de Valparaíso
**Autor:** Sebastián Inostroza Hurtado
**Año:** 2026

---

## Descripción

El cáncer de mama es la principal causa de mortalidad oncológica femenina en Chile (DEIS, 2024). Los informes mamográficos contienen información crítica para el seguimiento de las pacientes (categoría BI-RADS, recomendación clínica), pero al estar redactados en lenguaje natural quedan invisibles para los sistemas administrativos del centro asistencial.

Este proyecto desarrolla un **Módulo de Validación Técnica (MVP)** ejecutable localmente, sin transferir datos sensibles a la nube, capaz de:

1. **Predice** la categoría BI-RADS implícita de un informe mamográfico en español mediante NLP.
2. **Extrae** la categoría BI-RADS y la recomendación clínica declaradas explícitamente en el texto.
3. **Coteja** ambas con el estándar internacional BI-RADS/ACR para emitir alertas de inconsistencia.

La ejecución local garantiza que los datos clínicos no abandonen el centro asistencial, en cumplimiento de la **Ley N° 19.628** de Protección de la Vida Privada de Chile.

## Estado del proyecto

**Fase experimental completada**. Modelo seleccionado para el MVP: **DistilBETO + augmentación textual**.

| Métrica | Split único (test 872) | CV 5-fold |
|---|---|---|
| Accuracy | 0.9817 | 0.9844 ± 0.0032 |
| Macro F1 | 0.9386 | 0.8877 ± 0.0501 |
| Weighted F1 | 0.9822 | 0.9843 ± 0.0033 |

**Comparación contra el mejor modelo clásico** (LinearSVC balanced bajo CV: 0.7369 ± 0.0703): DistilBETO supera por +0.1508 en Macro F1, equivalente a 2.1σ del modelo clásico. Diferencia estadísticamente robusta.

**Hallazgo clínico clave**: en BI-RADS 4 (sospecha de malignidad, la categoría más sensible), DistilBETO detecta 10 de 11 casos (recall 0.91) vs 6 de 11 de LinearSVC (recall 0.55).

## Estructura del repositorio

```

proyecto-ia-mamografia/
├── data/
│   ├── raw/                          # Dataset original (Vázquez Noguera et al., 2025)
│   └── processed/
│       └── reports_cleaned.csv       # Corpus limpio (4357 informes)
│
├── notebooks/
│   ├── 01_limpieza.ipynb             # Carga, normalización NFKD, EDA
│   ├── 02_baseline_tfidf.ipynb       # 5 modelos clásicos con TF-IDF
│   ├── 03_transformers.ipynb         # DistilBERT inglés (3 variantes)
│   ├── 04_distilbeto.ipynb           # DistilBETO español + augmentación
│   ├── 04b_cv_distilbeto.ipynb       # CV 5-fold con augmentación per-fold
│   └── anexos/                       # Salidas reproducibles
│       ├── README.md
│       ├── confmat_.png             # Matrices de confusión normalizadas
│       ├── report_.csv              # Classification reports por clase
│       └── predicciones_distilbeto.json
│
├── .gitignore                        # Excluye .venv, checkpoints, archivos pesados
├── README.md                         # Este archivo
└── requirements.txt                  # Dependencias del proyecto (144 paquetes)

```

---
## Modelos evaluados

| Modelo | Accuracy | Macro F1 | Familia |
|---|---|---|---|
| **DistilBETO + aug (español)** | **0.9817** | **0.9386** | Transformer |
| LinearSVC balanced | 0.9828 | 0.9367 | Clásica (TF-IDF) |
| LinearSVC | 0.9794 | 0.8997 | Clásica (TF-IDF) |
| Logistic Regression balanced | 0.9472 | 0.8216 | Clásica (TF-IDF) |
| DistilBERT base (inglés) | 0.9461 | 0.4853 | Transformer |
| DistilBERT + aug (inglés) | 0.9323 | 0.4710 | Transformer |
| DistilBERT + class weights | 0.9335 | 0.4384 | Transformer |
| Logistic Regression | 0.9576 | 0.5223 | Clásica (TF-IDF) |
| Multinomial Naive Bayes | 0.8830 | 0.4169 | Clásica (TF-IDF) |

## Dataset

- **Origen**: Vázquez Noguera, J. L., et al. (2025). *Mammography reporting dataset with BI-RADS system for natural language processing applications*. Zenodo. DOI: 10.5281/zenodo.14827680
- **Tamaño**: 4 357 informes mamográficos en español, anonimizados.
- **Etiquetas**: categorías BI-RADS 0-6 asignadas por radiólogos.
- **Desbalance**: predominio de clases 1 y 2 (74.2%); BI-RADS 5 con 16 informes y BI-RADS 6 con solo 5.

## Reproducibilidad

### Requisitos

- Python 3.9
- macOS con Apple Silicon (MPS) o Linux con CUDA, recomendable para fine-tuning del Transformer
- ~5 GB de espacio en disco para checkpoints durante entrenamiento

### Instalación

```bash
git clone https://github.com/SebasDCIS/proyecto-ia-mamografia.git
cd proyecto-ia-mamografia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecución de los experimentos

Los notebooks deben ejecutarse en orden:

1. `01_limpieza.ipynb` - genera `data/processed/reports_cleaned.csv`
2. `02_baseline_tfidf.ipynb` - modelos clásicos sobre TF-IDF
3. `03_transformers.ipynb` - DistilBERT inglés
4. `04_distilbeto.ipynb` - DistilBETO español (modelo final)
5. `04b_cv_distilbeto.ipynb` - validación cruzada del modelo final (~25 min)

## Decisiones técnicas clave

### Por qué Macro F1 como métrica principal

El dataset está fuertemente desbalanceado. Accuracy es engañoso: un modelo que predijera siempre "BI-RADS 2" tendría ~60% de Accuracy sin aprender nada útil. Macro F1 promedia el F1 de cada clase con igual peso, garantizando que el modelo funcione bien también en las clases minoritarias (que son justamente las clínicamente críticas).

### Por qué DistilBETO sobre LinearSVC

Aunque ambos modelos tienen Macro F1 cercano a 0.94 en el split único, la validación cruzada de 5-fold confirma que DistilBETO mantiene una ventaja robusta de +0.15 en Macro F1 sobre LinearSVC. Además, DistilBETO obtiene mejor recall en BI-RADS 4 (sospecha de malignidad): detecta 10 de 11 casos vs 6 de 11 de LinearSVC, lo que en un sistema de validación clínica es decisivo.

### Por qué augmentación per-fold en la CV

Si la augmentación textual se aplica antes de hacer la CV, las variantes de un mismo informe podrían quedar repartidas entre train y test del mismo fold, produciendo data leakage. Aplicar la augmentación solo dentro del train de cada fold (después de la partición) garantiza una evaluación honesta.

## Pendiente para entrega final

- `src/extractor_birads.py`: extracción regex de BI-RADS declarado (con subcategorías 4a/4b/4c)
- `src/extractor_recomendacion.py`: segmentación del bloque de recomendaciones
- `src/cotejo_acr.py`: tabla normativa BI-RADS → recomendación esperada
- `src/comparador_semantico.py`: comparación semántica por capas (normalización, reglas, TF-IDF, decisión)
- `src/predict.py`: orquestador end-to-end que carga DistilBETO y entrega salida estructurada con alertas
- Conjunto de 10-20 casos de prueba sintéticos y reales para validar el sistema integrado

## Referencias principales

- Vázquez Noguera, J. L., et al. (2025). Mammography reporting dataset with BI-RADS system. *Data in Brief*.
- López-Úbeda, P., et al. (2024). Automatic classification and prioritisation of actionable BI-RADS categories. *Clinical Radiology*.
- Cañete, J., Chaperon, G., Fuentes, R., Ho, J. H., Kang, H., & Pérez, J. (2020). Spanish Pre-Trained BERT Model and Evaluation Data. *PML4DC at ICLR 2020*.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
- American College of Radiology. (2013). *BI-RADS® Atlas — Breast Imaging Reporting and Data System* (5.ª ed.).

## Licencia

Este proyecto es de uso académico, en el marco de la asignatura BME513. El código se publica bajo licencia MIT. El dataset original mantiene su licencia propia (consultar Zenodo).

