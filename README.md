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

Pipeline híbrido: reglas transparentes como vía primaria, IA como respaldo.

```
Informe → M0 Limpieza → M1 BI-RADS (regex+híbrido) → M2 Recomendación (reglas+sinónimos)
                              │                              │
                          Apoyo ML                      Respaldo NER
                        (DistilBETO)                   (DistilBETO)
                              └──────────────┬───────────────┘
                                    M3 Cotejo ACR → coherente / alerta / revisión
```

Tres capas de flexibilidad léxica, cada una en su rol:

- **Typos** (Damerau-Levenshtein): errores de tipeo en los verbos gatillo.
- **Sinónimos clínicos**: términos equivalentes (ultrasonido≈ecografía,
  seguimiento≈control, BACAF≈biopsia) normalizados antes de clasificar.
- **NER** (DistilBETO): localiza la recomendación en redacciones no vistas
  cuando las reglas no la encuentran.

## Estructura del repositorio

```
src/
  extractor_birads.py         Extracción de la categoría BI-RADS
  buscador_birads.py          Búsqueda híbrida (informes sin encabezado)
  extractor_recomendacion.py  Extracción y clasificación por reglas
  extractor_ner.py            Extractor NER de respaldo (DistilBETO)
  cotejo_acr.py               Motor de cotejo BI-RADS/ACR
  buscador_birads.py          Búsqueda híbrida del BI-RADS en 4 fases (la vía en uso)
  proto_typos_birads.py       Tolerancia a errores de tipeo en el token BI-RADS
  verificador_birads_ml.py    Verificador DistilBETO. DESACTIVADO: se midió que no
                              aporta sobre la vía reglada (ver docs/BITACORA.md)
  predict.py                  Orquestación del pipeline end-to-end
  recursos/
    vocabulario_clinico.py    Categorías, patrones, typos, sinónimos
    limpieza_informe.py       Limpieza de descargo/firma/datos del paciente
    tabla_acr.py              Tabla normativa BI-RADS/ACR
dashboard/                    Interfaz Streamlit
notebooks/                    Entrenamiento y evaluación del NER y del verificador
report/                       Informe LaTeX + figuras + PDF
docs/                         Defensa, validación clínica, guía teórica
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
abordaron mediante preprocesamiento, sinónimos y el extractor NER. La mejora de
fondo requiere un corpus del contexto de despliegue.

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

El Módulo 2 (recomendación) **sí** usa IA, porque su variación es semántica:
los formatos de un número se pueden enumerar en una tabla; las redacciones de
una recomendación clínica, no.

La cronología completa está en [`docs/BITACORA.md`](docs/BITACORA.md).
