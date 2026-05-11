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

1. **Predecir** automáticamente la categoría BI-RADS implícita en un informe mediante un modelo de NLP.
2. **Extraer** la categoría BI-RADS y la recomendación declaradas explícitamente en el texto.
3. **Contrastar** ambas con el estándar BI-RADS/ACR para detectar discrepancias y emitir alertas bajo un esquema *Human-on-the-Loop*.

La ejecución local cumple con la Ley N.º 19.628 sobre Protección de la Vida Privada y evita la exposición de información clínica a servicios externos.

---

## Estructura del repositorio

```

proyecto-ia-mamografia/
├── data/
│   ├── raw/                       # Corpus original (Vázquez Noguera et al., 2025)
│   └── processed/                 # Corpus depurado (reports_cleaned.csv)
├── notebooks/
│   ├── 01_limpieza.ipynb          # Limpieza y normalización del corpus
│   ├── 02_baseline_tfidf.ipynb    # Modelos clásicos: TF-IDF + LinearSVC
│   ├── 03_transformers.ipynb      # Fine-tuning de DistilBERT-es
│   ├── comparacion_modelos.csv    # Resultados consolidados
│   └── comparacion_modelos.png    # Gráfico comparativo
├── src/                           # (en construcción) Módulos del MVP integrado
├── .gitignore
└── README.md

```

---

## Dataset

Corpus público **Mammography reporting dataset with BI-RADS system for natural language processing** (Vázquez Noguera et al., 2025), disponible en Zenodo bajo DOI [10.5281/zenodo.14827680](https://doi.org/10.5281/zenodo.14827680).

- 4 357 informes mamográficos reales en español, anonimizados.
- Etiqueta BI-RADS en el rango 0–6 asignada por radiólogos.
- Marcado desbalance de clases: predominan las categorías 1 y 2 (negativo/benigno).

---

## Estado de avance

### Fase experimental — cerrada

| Componente | Estado |
|------------|--------|
| Limpieza del corpus (`01_limpieza.ipynb`) | ✅ |
| Baseline TF-IDF + LinearSVC balanceado | ✅ |
| Fine-tuning DistilBERT-es (base / class weights / augmentación) | ✅ |
| Validación cruzada estratificada del modelo ganador | ✅ |
| Selección de modelo base del MVP: **LinearSVC balanced** | ✅ |

### Resultados de selección

| Modelo | Accuracy | Macro F1 | Weighted F1 |
|--------|:--:|:--:|:--:|
| LinearSVC balanced (test 80/20) | 0.9472 | **0.8216** | 0.9419 |
| LinearSVC balanced (CV 5-fold) | — | 0.7369 ± 0.0703 | — |
| DistilBERT-es + augmentación | 0.9427 | 0.5738 | 0.9419 |

El modelo clásico fue seleccionado por mejor Macro F1, menor costo computacional (CPU local) y mayor explicabilidad clínica.

### MVP integrado — en construcción

| Componente | Estado |
|------------|--------|
| `src/extractor_birads.py` — extracción regex del BI-RADS declarado | ⏳ |
| `src/extractor_recomendacion.py` — segmentación de la recomendación | ⏳ |
| `src/cotejo_acr.py` — motor de cotejo normativo BI-RADS/ACR | ⏳ |
| `src/comparador_semantico.py` — comparación semántica por capas | ⏳ |
| `src/predict.py` — pipeline integrador end-to-end | ⏳ |

---

## Arquitectura del MVP


```

Informe (.txt)
│
▼
[1] Preprocesamiento
│
▼
[2] Predicción BI-RADS (LinearSVC)
│
▼
[3] Extracción BI-RADS y recomendación declarados (regex)
│
▼
[4] Motor de cotejo normativo BI-RADS/ACR
│
▼
Salida estructurada:
birads_predicho, birads_declarado,
recom_observada, recom_esperada,
alerta_categoria, alerta_recomendacion


```

---

## Reproducir el entorno

```bash
git clone https://github.com/SebasDCIS/proyecto-ia-mamografia.git
cd proyecto-ia-mamografia

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Para abrir los notebooks:

```bash
jupyter lab
```

---

## Referencias principales

- Vázquez Noguera, J. L., et al. (2025). *Mammography reporting dataset with BI-RADS system for natural language processing applications.* Data in Brief. [DOI](https://doi.org/10.1016/j.dib.2025.111761)
- López-Úbeda, P., et al. (2024). *Automatic classification and prioritisation of actionable BI-RADS categories using natural language processing models.* Clinical Radiology. [DOI](https://doi.org/10.1016/j.crad.2023.09.009)
- American College of Radiology. (2013). *BI-RADS® Atlas — Breast Imaging Reporting and Data System* (5.ª ed.).

---

## Licencia y privacidad

Proyecto académico. No procesa datos clínicos identificables. El corpus utilizado es público y anonimizado en origen. Cualquier uso futuro sobre datos reales debe realizarse bajo los protocolos de privacidad de la institución y la Ley N.º 19.628.

