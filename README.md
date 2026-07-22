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

## Uso

```bash
pip install -r requirements.txt

# Pipeline por línea de comandos (suites de prueba de cada módulo)
python -m src.predict

# Interfaz web
streamlit run dashboard/app.py
```

El extractor NER requiere el modelo entrenado en `models/ner_recomendacion_final`
(ver `notebooks/11_extractor_ner_recomendacion.ipynb`). Si el modelo no está
presente, el sistema opera solo con reglas sin interrumpirse.

## Datos y limitación

Corpus de entrenamiento: 4 357 informes en español (Vázquez Noguera et al., 2025),
de **origen paraguayo**. El contexto de despliegue previsto es chileno. Esta
diferencia es una limitación explícita: el corpus es homogéneo y no contiene los
formatos, descargos ni variantes léxicas de los informes chilenos, que se
abordaron mediante preprocesamiento, sinónimos y el extractor NER, y se validaron
contra 19 informes reales (ver más abajo). La mejora de fondo sigue requiriendo un
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

## Validación con informes reales

El sistema se probó sobre **19 informes chilenos reales** (Bupa Clínica Reñaca,
marzo de 2026), distintos en formato al corpus de entrenamiento.

| | |
|---|---|
| BI-RADS extraído | 19/19, todos con confianza alta |
| Recomendación clasificada | 14/19 (los otros 5 no la declaran) |
| Fugas de datos personales | 0 en todas las vistas |
| Alertas generadas | 4 de severidad media, 1 crítica |

Dos de esos informes escriben la categoría con **numerales romanos**
(`Birads -us III`), forma que el corpus paraguayo no contiene en absoluto. El
soporte para romanos se había programado como cobertura preventiva, sin disponer
de un solo ejemplo, y aquí demostró su valor.

La validación destapó seis fallos que el corpus no revelaba: datos personales
pegados al texto clínico que sobrevivían a la limpieza, el dashboard mostrando el
texto crudo, un subjuntivo descriptivo confundido con un verbo gatillo, el título
del examen leído como recomendación, la forma verbal de "control" no reconocida, y
una comparación regex/NER que reportaba concordancia donde no la había. Todos
corregidos y verificados sin regresión. El detalle está en
[`docs/BITACORA.md`](docs/BITACORA.md).

Uno de los informes es un BI-RADS 5 con neoplasia y adenopatías metastásicas que
**no declara ninguna recomendación**. A raíz de ese caso, la severidad escala a
crítica cuando el BI-RADS es 4, 5 o 6 y falta la conducta: es la contraparte de la
alerta de omisión del Módulo 1.

Estos 19 informes permiten **detectar fallos**, no medir desempeño: no están
anotados y provienen de un solo centro.

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
