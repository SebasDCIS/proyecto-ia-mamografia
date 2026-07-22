# Auditoría automatizada de informes mamográficos

Sistema local e interpretable para auditar la coherencia entre la categoría
**BI-RADS** declarada y la **recomendación clínica** en informes mamográficos en
español, según el estándar BI-RADS/ACR.

**Proyecto BME513 · Universidad de Valparaíso** — Sebastián Inostroza Hurtado
Supervisor: Dr. Alejandro Veloz Baeza.

---

## Qué hace

Dado un informe mamográfico (texto o PDF), el sistema:

1. Limpia el informe (descargo legal, firma, datos del paciente).
2. Extrae la categoría BI-RADS declarada.
3. Extrae y clasifica la recomendación clínica.
4. Coteja ambas contra la tabla normativa ACR y reporta si son coherentes,
   incoherentes (con severidad) o si el caso requiere revisión humana.

El sistema **no decide la conducta clínica**: detecta inconsistencias y las
deriva a revisión (filosofía *human-on-the-loop*).

## Arquitectura

Pipeline de reglas transparentes, con IA solo en el módulo donde la evidencia
respalda su uso.

```
Informe → M0 Limpieza → M1 BI-RADS (solo reglas) → M2 Recomendación (reglas+sinónimos)
                              │                              │
                        Sin IA: el ML                   Respaldo NER
                       no aportó (medido)               (DistilBETO)
                              └──────────────┬───────────────┘
                                    M3 Cotejo ACR → coherente / alerta / revisión
```

El **M1 opera solo con reglas** (búsqueda híbrida en 4 fases, Macro F1 = 0,9995).
Se entrenó un verificador DistilBETO para contrastarlo y se retiró tras medir que
no aporta; el detalle está más abajo y en [`docs/BITACORA.md`](docs/BITACORA.md).

Tres capas de flexibilidad léxica, cada una en su rol:

- **Typos** (Damerau-Levenshtein): errores de tipeo en los verbos gatillo del M2,
  y en el token `birads` del M1. Nunca sobre los dígitos: `BI-RADS 2` y `BI-RADS 5`
  están a distancia 1 y significan cosas opuestas.
- **Sinónimos clínicos**: términos equivalentes (ultrasonido≈ecografía,
  seguimiento≈control, BACAF≈biopsia) normalizados antes de clasificar.
- **NER** (DistilBETO): localiza la recomendación en redacciones no vistas
  cuando las reglas no la encuentran.

## Estructura del repositorio

```
src/
  buscador_birads.py          M1: búsqueda híbrida del BI-RADS en 4 fases (la vía en uso)
  extractor_birads.py         M1: versión previa, dependía del encabezado CONCLUSIÓN
  proto_typos_birads.py       M1: tolerancia a typos en el token BI-RADS (prototipo)
  extractor_recomendacion.py  M2: extracción y clasificación por reglas (TF-IDF)
  extractor_ner.py            M2: extractor NER de respaldo (DistilBETO)
  cotejo_acr.py               M3: motor de cotejo BI-RADS/ACR
  verificador_birads_ml.py    M4: DESACTIVADO. Se midió que no aporta sobre la vía
                              reglada; se conserva con su evaluación (ver BITACORA)
  predict.py                  Orquestación del pipeline end-to-end
  recursos/
    vocabulario_clinico.py    Categorías, patrones, typos, sinónimos
    limpieza_informe.py       M0: limpieza de descargo/firma/datos del paciente
    tabla_acr.py              Tabla normativa BI-RADS/ACR
dashboard/                    Interfaz Streamlit
notebooks/                    Exploración, entrenamiento y evaluación (19 notebooks)
  04c_cv_ventana_local        CV del verificador sobre su ventana real de producción
  11_extractor_ner            Entrenamiento del NER
  11b_ablacion_ner            Ablación: ¿el NER lee el encabezado o el contenido?
  11c_estres_ner              Prueba de estrés: redacciones no anticipadas
  *_Colab                     Experimentos en GPU (embeddings, predicción BI-RADS)
report/                       Informe LaTeX + figuras + PDF
docs/
  BITACORA.md                 Cronología del proyecto y las mediciones que lo guiaron
  Presentacion_BME513         Presentación de defensa
  Guia_Fundamentos_IA         Guía de conceptos de IA aplicados al proyecto
```

## Requisitos

- **Python 3.9** o superior
- Sistema operativo: probado en macOS (Apple Silicon, aceleración MPS) y Linux
- Espacio en disco: ~1 GB si se descarga el modelo NER entrenado

Las dependencias están en [`requirements.txt`](requirements.txt). El núcleo del
sistema (extracción y cotejo por reglas) solo necesita `numpy` y `scikit-learn`;
`torch` y `transformers` se requieren únicamente para el extractor NER y los
notebooks de experimentación, y `streamlit` para la interfaz.

## Instalación

```bash
git clone https://github.com/SebasDCIS/proyecto-ia-mamografia.git
cd proyecto-ia-mamografia

python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Obtención de los datos

El corpus **no se incluye en el repositorio** (está en `.gitignore`). Es público y
se descarga desde Zenodo:

- **Corpus:** *Mammography reporting dataset with BI-RADS system for natural
  language processing applications*, Vázquez Noguera et al. (2025)
- **Zenodo:** [10.5281/zenodo.14827680](https://doi.org/10.5281/zenodo.14827680)
- **Artículo:** [10.1016/j.dib.2025.111761](https://doi.org/10.1016/j.dib.2025.111761)
- **Contenido:** 4 357 informes mamográficos en español, con la categoría BI-RADS
  y la recomendación clínica en columnas separadas

Tras descargarlo, colocar el CSV en:

```
data/processed/reports_cleaned.csv
```

Los notebooks que preparan el corpus a partir del archivo original son
`notebooks/00_exploracion_informes.ipynb` y `notebooks/01_limpieza.ipynb`.

## Ejecución

**Sin necesidad de descargar datos** (usa casos de prueba incorporados):

```bash
# Suite de pruebas del pipeline completo: 8 casos
python -m src.predict

# Batería de cobertura de formatos: 16 casos sintéticos
python -m tests.casos_formato_chileno
```

**Procesar un informe concreto:**

```bash
python -m src.predict --input ruta/al/informe.txt
python -m src.predict --input ruta/al/informe.pdf --type pdf
python -m src.predict --input informe.txt --output resultado.json
```

**Interfaz web:**

```bash
streamlit run dashboard/app.py
```

El extractor NER requiere el modelo entrenado en `models/ner_recomendacion_final`
(se genera con `notebooks/11_extractor_ner_recomendacion.ipynb`). Si el modelo no
está presente, el sistema opera solo con reglas sin interrumpirse.

## Reproducir los experimentos

Los notebooks están numerados en el orden en que se ejecutaron. Requieren el
corpus descargado.

| Notebook | Qué reproduce | Requiere GPU |
|---|---|---|
| `00`, `01` | Exploración del corpus y limpieza | no |
| `02_baseline_tfidf` | Líneas base clásicas (LinearSVC, NB, LogReg) | no |
| `03_transformers` | DistilBERT en inglés: Macro F1 = 0,471 | recomendada |
| `04_distilbeto` | Mismo experimento en español: 0,9386 | recomendada |
| `04b_cv_distilbeto` | Validación cruzada. Revela la fuga por aumentación | recomendada |
| `04c_cv_ventana_local` | CV del verificador sobre su ventana real de inferencia | recomendada |
| `05_extractor_birads` | Extractor reglado: Macro F1 = 0,9995 | no |
| `07_validacion_cotejo_acr` | Validación end-to-end del cotejo | no |
| `08_verificador_birads_ml` | Verificador Transformer (después retirado) | recomendada |
| `11_extractor_ner` | Entrenamiento del NER: F1 de span = 0,9991 | recomendada |
| `11b_ablacion_ner` | Ablación: ¿el NER lee el encabezado o el contenido? | recomendada |
| `11c_estres_ner` | Prueba de estrés con redacciones no anticipadas | recomendada |
| `*_Colab` | Embeddings y predicción de BI-RADS (ejecutados en Colab) | sí |

**Reproducción mínima sin GPU ni descarga de datos:** las dos suites de la
sección anterior (`src.predict` y `tests.casos_formato_chileno`) verifican el
pipeline reglado completo y la cobertura de formatos, que son las vías que operan
en producción.

## Limitación del corpus

Corpus de entrenamiento: 4 357 informes en español (Vázquez Noguera et al., 2025),
de **origen paraguayo**. El contexto de despliegue previsto es chileno. Esta
diferencia es una limitación explícita: el corpus es homogéneo y no contiene los
formatos, descargos ni variantes léxicas de los informes chilenos, que se
abordaron mediante preprocesamiento, sinónimos y el extractor NER, con una batería
de casos de prueba que cubre esas variantes (ver más abajo). La mejora de fondo sigue requiriendo un
corpus anotado del contexto de despliegue.

## Privacidad

Todo el procesamiento es local; ningún dato se envía a servicios externos. La
capa de limpieza retira identificadores del paciente antes del análisis, en
concordancia con la Ley N.º 19.628.

---

## Sobre el rol de la IA en este sistema

El Módulo 1 (BI-RADS) opera **solo con reglas**, con Macro F1 = 0,9995. Se
entrenó un verificador DistilBETO para contrastarlo y cuatro mediciones
independientes mostraron que no aporta:

- **Ablación**: al enmascarar el número declarado, el Macro F1 cae de 0,939 a 0,544. El modelo lee la categoría, no la deduce.
- **Frecuencia**: el arbitraje no se activa en ninguno de los 4 357 informes (el 99,9 % declara una sola mención).
- **Línea base trivial**: sobre la ventana de 250 caracteres que el modelo recibe, un `re.search` de una línea alcanza 99,82 % y DistilBETO 99,78 %.
- **Formatos no vistos**: los 2 informes con typo se recuperan con tolerancia de edición (2/2, sin falsos positivos); el modelo no los recupera.

La causa es estructural: la ventana se construye alrededor de la mención que la
regla seleccionó, así que ambas leen la misma cadena. No es una segunda opinión
sino una relectura. Leer una categoría declarada no admite discrepancia
informativa.

### El NER se sometió al mismo estándar

Descartar un modelo con la ablación obliga a aplicarle esa misma prueba al otro.

Primero, la parte incómoda: **dentro del corpus el NER tampoco aporta.** Una
expresión regular que localiza el último verbo gatillo y marca hasta el final
alcanza F1 = 0,9991 frente al 1,0000 del NER. El corpus es homogéneo: el 98,1 %
de las recomendaciones empieza con la misma fórmula y todas ocupan la posición
final.

La diferencia está en el **rol**. El verificador actuaba dentro del corpus, así
que la métrica del corpus era la relevante y mostró que no servía. El NER es un
respaldo para casos **fuera** del corpus, de modo que esa métrica no evalúa su
función. Como la evidencia externa era acotada, se simuló la brecha
sobre los 565 informes de prueba: se quitó el encabezado y se reemplazó el verbo
gatillo por formas verificadas como ausentes de la lista cerrada del módulo.

| Condición del informe | Regla | NER |
|---|---|---|
| Control | 1,0000 | 1,0000 |
| Sin encabezado `RECOMENDACIONES` | 0,9460 | 1,0000 |
| Verbo fuera de la lista de reglas | 1,0000 | 0,9929 |
| **Sin encabezado y verbo nuevo** | **0,5314** | **0,9956** |

La regla dispone de dos señales y se sostiene mientras conserve una: sin
encabezado la salva el verbo, con un verbo desconocido la salva el encabezado.
Al retirar ambas queda en 0,5314. El NER no depende de ninguna.

El contraste bajo la misma prueba resume la decisión de diseño:

| Modelo | Al quitarle su señal | |
|---|---|---|
| Verificador BI-RADS | 0,939 → **0,544** | colapsa · se retiró |
| NER de recomendación | 1,000 → **0,9956** | aguanta · se conservó |

El experimento es **sintético**: opera sobre el corpus paraguayo perturbado, no
sobre informes chilenos reales. Mide el rol que el componente cumple, no su
desempeño en Chile.

## Validación de cobertura de formatos

El corpus de entrenamiento es homogéneo: encabezados consistentes, sin numerales
romanos, ya anonimizado, con la recomendación siempre en la misma posición. Esa
homogeneidad permite que el sistema sostenga supuestos sin costo aparente.

Para verificar la cobertura sobre variantes que el corpus no contiene se construyó
una **batería de 16 casos de prueba sintéticos**
([`tests/casos_formato_chileno.py`](tests/casos_formato_chileno.py)). Son informes
ficticios: nombres, identificadores y fechas inventados. Cubren variantes de
redacción y estructura documentadas en la práctica clínica local.

| Grupo | Variantes cubiertas |
|---|---|
| Escritura del BI-RADS | Arábigo, numeral romano (`Birads -us III`), modalidad antepuesta (`US BIRADS 1`) y pospuesta (`Birads 2 US`), error de tipeo (`bi-radas`) |
| Estructura | `Impresión mamográfica` en vez de `Conclusión`, sin encabezado, descargo legal reubicado al inicio, mención histórica previa a la definitiva |
| Recomendación | Forma verbal (`controlar` en vez de `control`), sinónimos de técnica (`ultrasonido`) |
| Comportamiento seguro | Hallazgos sin categoría, sospecha sin conducta declarada, incoherencia crítica |
| Privacidad | Nombre e identificador pegados al texto clínico |

Resultado: **16/16** en las cuatro dimensiones evaluadas (categoría extraída,
recomendación clasificada, estado del cotejo, ausencia de identificadores tras la
limpieza). La batería es ejecutable sin acceso a datos clínicos:

```bash
python -m tests.casos_formato_chileno
```

Uno de los casos es un BI-RADS 5 que no declara ninguna recomendación. A raíz de
ese escenario, la severidad escala a crítica cuando el BI-RADS es 4, 5 o 6 y falta
la conducta: es la contraparte de la alerta de omisión del Módulo 1.

Estos casos verifican **cobertura de formatos**, no desempeño poblacional. La
validación con un corpus clínico anotado del contexto de despliegue sigue siendo
la limitación de fondo del trabajo.

## Resultados

| Componente | Métrica | Valor |
|---|---|---|
| M1 · Extracción BI-RADS | Macro F1 | **0,9995** |
| M1 · Confianza alta | % del corpus | 98,58 % |
| M2 · Cobertura de recomendación | % del corpus | 99,82 % |
| M2 · NER | F1 de span (test deduplicado) | **0,9991** |
| M3 · Alertas de incoherencia | sobre 4 357 informes | 50 (1,15 %), 19 críticas |

Sobre las decisiones de modelado: el idioma del preentrenamiento explica una
brecha de 0,471 (DistilBERT en inglés) a 0,939 (DistilBETO en español) en la
misma tarea. La validación cruzada reveló una fuga por aumentar antes de
particionar, que corrigió el desempeño de 0,9386 a 0,8877. Predecir el BI-RADS
desde los hallazgos alcanza Macro F1 = 0,624 fuera de fold, insuficiente en las
clases críticas: el BI-RADS 6 tiene cinco ejemplos en todo el corpus.

La cronología completa, con las mediciones que guiaron cada decisión, está en
[`docs/BITACORA.md`](docs/BITACORA.md).
