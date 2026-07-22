# Bitácora del proyecto

Proyecto BME513 · Universidad de Valparaíso · Sebastián Inostroza Hurtado
Supervisor: Dr. Alejandro Veloz Baeza

Este documento registra las decisiones de diseño y las mediciones que las
determinaron. Cada entrada indica qué se probó, qué se midió y qué se decidió.
Las cifras provienen de los notebooks del directorio `notebooks/` y son
reproducibles.

---

## Línea de tiempo

Las fechas provienen del historial de commits del repositorio. Corresponden a
cuándo se registró cada avance, que puede ser posterior al día en que se hizo el
trabajo.

| Fecha | Hito |
|---|---|
| 28-abr-2026 | Inicio del proyecto |
| 07-may · 11-may | Exploración del corpus, limpieza, baselines clásicos (nb 00 a nb 02) |
| 12-may-2026 | **nb 04: DistilBETO con aumentación textual.** Se decide adoptarlo como modelo del MVP |
| 16-may-2026 | **nb 04b: CV 5-fold.** Aparece la fuga por aumentar antes de particionar: 0,9386 → 0,8877 |
| 18-may-2026 | **nb 05: extractor de BI-RADS declarado.** Se identifica que la tarea correcta es extraer, no predecir |
| 31-may-2026 | `extractor_birads.py` pasa a módulo importable |
| 04-jun-2026 | **nb 06 y nb 07: extractor de recomendaciones y cotejo ACR.** MVP validado end-to-end |
| 19-jun-2026 | **nb 08: el modelo pasa a verificador secundario** (Módulo 4, lógica v2.1) |
| 20-jun-2026 | Integración del verificador al cotejo (v2), orquestador `predict.py`, dashboard Streamlit, informe LaTeX v8 |
| 27-jun-2026 | Informe v8.1: se reporta el 0,8877 de CV como métrica honesta |
| — | Entrenamiento del NER (nb 11) y rediseño del buscador tras los informes chilenos |
| **21-jul-2026** | **El Módulo 1 pasa a solo reglas.** Se retira el verificador tras cuatro mediciones. El NER se somete al mismo estándar (nb 11b, nb 11c) |
| **21-jul-2026** | **Batería de 16 casos de formato.** Seis correcciones: redacción intra-línea de identificadores, texto crudo en el dashboard, dos falsos positivos de extracción, la forma verbal de "control", y una concordancia regex/NER que mentía. Escalamiento a severidad crítica por conducta ausente |

---

## Los modelos clásicos primero (nb 01, nb 02) · mayo 2026

Antes de cualquier red neuronal se establecieron líneas base. La tarea era
clasificar el informe en una de las siete categorías BI-RADS.

| Modelo | Macro F1 |
|---|---|
| MultinomialNB | 0,42 |
| Regresión logística | 0,52 |
| LinearSVC | 0,90 |
| LinearSVC `class_weight='balanced'` | **0,9367** |

LinearSVC balanced quedó como referencia a superar, validado con CV 5-fold
(0,7369 ± 0,0703).

---

## El Transformer en inglés, y el sesgo que introduje (nb 03) · mayo 2026

Se entrenó `distilbert-base-uncased` sobre el corpus. **Macro F1 = 0,471**,
con exactitud de 0,932.

Esa brecha entre exactitud y Macro F1 es el hallazgo. El modelo acertaba las
clases abundantes por fuerza bruta estadística y se derrumbaba en las raras.
Con 2 635 ejemplos de BI-RADS 2 se puede memorizar un patrón sin entender el
idioma; con 16 ejemplos de BI-RADS 5, no.

El problema: `uncased` sin más apellido es la versión en **inglés**. Se le
pidió a un modelo entrenado en inglés que leyera español. El vocabulario del
tokenizer, construido sobre Wikipedia en inglés, fragmenta las palabras en
español en pedazos sin sentido.

---

## Aislar la variable idioma (nb 04) · 12-may-2026

Para separar "el Transformer no sirve aquí" de "le pedí leer un idioma que no
conoce", se replicó el experimento del nb 03 cambiando **solo** el modelo base
a `dccuchile/distilbert-base-spanish-uncased` (DistilBETO). Mismo corpus, mismo
split (`random_state=42`), misma aumentación, mismos hiperparámetros.

**Macro F1 = 0,9386.** El idioma era la causa.

Con ese resultado se decidió hacer de DistilBETO el modelo base del MVP, en
reemplazo de LinearSVC. La ventaja se concentraba en BI-RADS 4 (F1 = 0,7692),
la clase donde LinearSVC más se confundía.

---

## La fuga por aumentación (nb 04b) · 16-may-2026

Antes de cerrar la decisión anterior se validó DistilBETO con CV 5-fold, como
ya se había hecho con LinearSVC. El criterio se fijó **antes** de correr el
experimento: si el desempeño se mantenía cerca de 0,93 la elección quedaba
firme; si caía con fuerza, había que reconsiderarla.

Al preparar la validación apareció una fuga de datos. La aumentación textual
genera variantes léxicas de un mismo informe. Si se aumenta **antes** de
particionar, el informe original queda en entrenamiento y su variante en
prueba: el modelo la reconoce en vez de resolverla, y la métrica se infla.

La corrección consiste en aumentar **dentro** de cada fold, solo sobre su
conjunto de entrenamiento.

| | Macro F1 |
|---|---|
| Aumentar y luego particionar (con fuga) | 0,9386 |
| Particionar y luego aumentar (limpio) | **0,8877 ± 0,0501** |

Cayó. Se reportó la caída.

El error no estaba en aumentar ni en particionar de forma estratificada: ambas
son buenas prácticas. Estaba en el orden. No hay mensaje de error ni
advertencia; el código corre y la métrica sale mejor de lo que corresponde.

---

## La tarea correcta no era la que estaba resolviendo (nb 05) · 18-may-2026

Al construir el extractor por reglas quedó clara una distinción que cambió el
proyecto: **el modelo predice una categoría; la regla extrae la que el
radiólogo declaró**. Son tareas distintas.

Para auditar coherencia hace falta la segunda. La regla alcanza **99,93 % de
exactitud (Macro F1 = 0,9995)**, contra el 0,8877 del modelo, porque no está
resolviendo el mismo problema.

Exploración del corpus previa al diseño:

- 100 % de los informes tienen encabezado de conclusión (`Conclusión` 67,7 %, `Valoración` 32,3 %)
- 99,95 % tienen al menos una mención BI-RADS detectable
- 2 informes tienen errores de tipeo que la ocultan
- 0 % usa números romanos o la categoría escrita en palabras

El modelo pasó entonces de candidato principal a verificador secundario (nb 08).

---

## El modelo pasa a verificador secundario (nb 08) · 19-jun-2026

Con la regla resolviendo la extracción, el Transformer quedó sin su rol original.
En lugar de descartarlo se le asignó uno nuevo: releer el mismo texto y contrastar
su lectura con la de la regla, bajo una jerarquía explícita.

> **Jerarquía de fuentes.** 1. Regla: lo que el radiólogo escribió literalmente
> (autoridad clínica). 2. Modelo: patrón estadístico aprendido (segunda opinión
> técnica).

El diseño priorizaba la regla y reservaba la voz del modelo para los casos de baja
confianza, distinguiendo cinco estados según el acuerdo entre ambas vías. Al día
siguiente (20-jun) se integró al motor de cotejo (versión 2), se cerró el
orquestador `predict.py` y se construyó el dashboard en Streamlit, que muestra el
BI-RADS extraído, la recomendación clasificada, el veredicto del cotejo, la regla
aplicada y el estado de la verificación.

El 27-jun el informe se actualizó para reportar el 0,8877 de validación cruzada en
lugar del 0,9386 del split único, como métrica honesta del modelo.

Esta arquitectura se mantuvo hasta la auditoría del 21 de julio.

---

## Las variantes de formato rompieron el extractor · junio-julio 2026

El extractor del nb 05 localiza el bloque de conclusión por su encabezado y
falla por completo si no lo encuentra. Sobre el corpus paraguayo eso cubre el
100 % de los informes, así que la dependencia era invisible desde dentro del
corpus.

Al contrastar el módulo con variantes de formato de la práctica clínica local
aparecieron estructuras que el corpus no contiene: `Impresión Diagnóstica` en
lugar de `Conclusión`, informes sin encabezado explícito, y menciones múltiples
de la categoría (una histórica y una definitiva).

La respuesta no fue agregar excepciones sino eliminar la dependencia:
`buscador_birads.py` recorre el informe completo en cuatro fases (regex
exhaustiva, filtrado contextual, ponderación posicional, arbitraje en caso de
empate). La ponderación posicional tiene fundamento clínico y medido: la
categoría definitiva vive en la conclusión, y la última mención se ubica en el
percentil 0,85 del texto en el 100 % de los informes del corpus.

---

## El NER, y la fuga otra vez (nb 11) · junio-julio 2026

Para la recomendación clínica el problema es distinto: la variación es
**semántica**, no tipográfica. Hay unas veinte formas de escribir un BI-RADS 4
y se pueden enumerar en una tabla; las formas de redactar una recomendación no
se pueden enumerar.

Se entrenó un NER DistilBETO con etiquetas derivadas automáticamente (la
columna de recomendación aparece textualmente dentro del informe, lo que
permite alinearla y generar etiquetas BIO sin trabajo manual).

La misma disciplina se aplicó desde el diseño: separación antes de tokenizar,
test tocado una sola vez, y **auditoría de duplicados antes de particionar**.
El corpus tiene informes repetidos y casi repetidos; deduplicando por firma se
eliminaron 580 de 4 345 (13,3 %).

**F1 de span = 0,9991** sobre test deduplicado.

Esa cifra mide la facilidad del corpus más que la capacidad de generalizar: la
recomendación está al final del informe en el 99 % de los casos y existen solo
221 recomendaciones distintas, una de ellas repetida 784 veces. La prueba real
fueron los informes chilenos, donde el NER localizó recomendaciones que las
reglas no anticiparon.

---

## El verificador no aportaba · 21-jul-2026

El verificador del nb 08 llevaba desde entonces en el pipeline como apoyo
acotado. Una auditoría motivada por métricas sospechosamente altas mostró que
no aporta. Cuatro mediciones independientes:

**1. Ablación.** Enmascarando el número BI-RADS del texto y reevaluando, el
Macro F1 cae de 0,939 a **0,544** (−42 %). El modelo lee la categoría
declarada; no la deduce de los hallazgos.

**2. Frecuencia de intervención.** El arbitraje del modelo (fase 4 del
buscador) no se activa en **ninguno de los 4 357 informes**, porque el 99,9 %
declara una sola mención. Además el verificador lo invoca explícitamente
desactivado (`usar_ml_si_ambiguo=False`).

**3. Comparación contra una línea base trivial.** En inferencia el modelo lee
una ventana de 250 caracteres centrada en la mención (`-200/+50`). Sobre esa
misma ventana:

| Método | Exactitud |
|---|---|
| `re.search` de una línea | **99,82 %** |
| DistilBETO (67 M parámetros) | 99,78 % |

La validación cruzada sobre ventanas deduplicadas da Macro F1 = 0,9958 ±
0,0043, frente a 0,8920 ± 0,0805 sobre el informe completo. El alza no indica
un mejor modelo: recortar el texto alrededor de la respuesta reduce el problema
a copiar un dígito.

**4. Recuperación de formatos no vistos.** Los dos informes cuyo BI-RADS la
regla no lee (`bi-rads o` con la letra o, `bi-radas 4`) se recuperan con
tolerancia de edición sobre el token `birads`, sin tolerancia alguna sobre el
dígito. Resultado sobre el corpus completo: 2 de 2 recuperados, 0 falsos
positivos, 0 regresiones. El modelo no puede hacerlo: el corpus de
entrenamiento no contiene ninguna variante tipográfica, de modo que sus errores
se correlacionan con los de la regla en lugar de compensarlos.

**Limitación estructural.** La ventana se construye alrededor de la mención que
la regla seleccionó, así que ambas vías leen la misma cadena. El contraste no
es una segunda opinión independiente sino una relectura. Si la regla eligiera
la mención equivocada, el modelo leería esa misma mención y coincidiría con
ella: la verificación no puede detectar el error más relevante de la
extracción.

**Causa de fondo.** Leer una categoría declarada no admite discrepancia
informativa. `BI-RADS 4` no es ambiguo, y dos métodos que lo leen no tienen
cómo discrepar salvo por ruido.

**Decisión.** `usar_verificador_ml=False`. El Módulo 1 opera solo con reglas.
El código y su evaluación se conservan porque el resultado negativo delimita el
rol de la IA en el sistema.

---

## La fuga en la propia validación cruzada (nb 04c) · 21-jul-2026

La auditoría anterior destapó que el nb 04b **nunca dedujo el corpus**. Su CV
tenía un 14,9 % de ejemplos de prueba con gemelo idéntico en entrenamiento. La
disciplina aplicada en el NER no se había aplicado al verificador.

Y al recortar a ventanas el problema se triplica: informes distintos con la
misma conclusión colapsan en ventanas idénticas. 487 duplicados a nivel de
informe se vuelven 1 649 a nivel de ventana, y el 44,6 % del test tiene gemelo
en entrenamiento. **La transformación misma crea duplicados que no existían.**

Al deduplicar y volver a medir, el 0,8877 subió a 0,8920: los duplicados no
estaban inflando. La preocupación quedó descartada con medición.

**Criterio de deduplicación.** En el nb 11 se deduplicó por una firma del texto
sin números, porque ahí la etiqueta es la posición de un span. Aquí la etiqueta
**es** el dígito, así que esa firma habría fusionado ventanas de BI-RADS
distintos. Verificación: firma sin dígitos produce 6 grupos con etiquetas
contradictorias; texto exacto produce 0. El criterio de deduplicación depende
de cuál es la etiqueta, y no se copia de un notebook a otro sin pensar.

---

## El NER bajo el mismo estándar (nb 11b, nb 11c) · 21-jul-2026

Si el verificador se descartó por cuatro mediciones, corresponde aplicarle al NER
las mismas pruebas. No hacerlo sería exigirle a un modelo lo que al otro se le
perdona.

**La línea base trivial.** Sobre el corpus, un regex que localiza el último verbo
gatillo y marca hasta el final alcanza F1 = 0,9991 frente al 1,0000 del NER.
**Sobre el corpus paraguayo el NER no aporta**, y por la misma razón que el
verificador: el corpus es homogéneo. El 98,1 % de las recomendaciones empieza con
la misma fórmula y todas ocupan la posición final.

**Pero el rol es distinto, y ahí está la diferencia.** El verificador actuaba
dentro del corpus, así que la métrica del corpus era la métrica relevante y mostró
que no servía. El NER es un respaldo para casos fuera del corpus, de modo que la
métrica del corpus no evalúa su rol. El 0,9991 nunca fue su defensa: solo prueba
que aprendió la tarea.

**La ablación** (nb 11b). Enmascarando el encabezado `RECOMENDACIONES` y
reevaluando el modelo ya entrenado: 1,0000 → 1,0000. No dependía del encabezado.

**La prueba de estrés** (nb 11c). Como la evidencia real eran entonces tres
informes chilenos, se simuló la brecha sobre los 565 informes de prueba: se quitó el
encabezado y se reemplazó el verbo gatillo por formas verificadas una a una como
ausentes de `_FRASES_GATILLO_RECOMENDACION`. La regla evaluada es el módulo de
producción en su segunda vía, la que corre ante un PDF sin columna de
recomendaciones.

| Condición | Regla | NER |
|---|---|---|
| Control | 1,0000 | 1,0000 |
| Sin encabezado | 0,9460 | 1,0000 |
| Verbo fuera de la lista | 1,0000 | 0,9929 |
| **Sin encabezado y verbo nuevo** | **0,5314** | **0,9956** |

La regla dispone de **dos señales** y se sostiene mientras conserve una: sin
encabezado la salva el verbo, con un verbo desconocido la salva el encabezado. Al
retirar ambas queda en 0,5314. El NER no depende de ninguna.

La última fila es el caso chileno: sin encabezado explícito y con una redacción no
prevista. La regla pierde cerca de la mitad de las recomendaciones; el NER conserva
0,9956.

**El contraste bajo la misma prueba:**

| Modelo | Al quitarle su señal |
|---|---|
| Verificador BI-RADS | 0,939 → **0,544** · colapsa |
| NER | 1,000 → **0,9956** · aguanta |

El verificador disponía de una sola señal, el número declarado, y sin ella no le
quedaba nada que leer. El NER aprendió a reconocer qué constituye una recomendación
en lugar de memorizar los términos que la introducen.

**Limitación.** El experimento es sintético: opera sobre el corpus paraguayo
perturbado, no sobre informes chilenos reales. Mide el rol que el componente
cumple, no su desempeño en Chile. La validación con un corpus chileno anotado sigue
siendo la limitación de fondo.

**Nota de proceso.** El nb 11b requirió cuatro correcciones antes de dar un
resultado válido: columnas equivocadas (`Full_Report` en vez de
`Full_Report_clean`, y el modelo es *uncased*), mapeo de etiquetas equivocado (el
nb 11 nunca pasó `id2label` al crear el modelo, así que quedó con `LABEL_0/1/2`), y
una función `limpiar_rec` reescrita de memoria en vez de copiada, que corría el
span un token y anulaba el 100 % de las coincidencias. Las tres primeras versiones
dieron F1 cercano a cero. La lección: las funciones de etiquetado se copian
textuales del notebook de entrenamiento, no se reescriben. Y una puerta de sanidad
que bloquee el veredicto cuando el control no reproduce la referencia es
obligatoria: sin ella, la primera versión llegó a imprimir un veredicto favorable
sobre un experimento roto.

**Deuda técnica detectada.** El nb 11 creó el modelo con
`AutoModelForTokenClassification.from_pretrained(MODELO, num_labels=3)`, sin pasar
`id2label`. No afecta sus métricas, porque el notebook decodifica con su propio
diccionario, pero cualquiera que cargue el modelo desde fuera obtiene etiquetas sin
significado. La corrección es pasar `id2label` y `label2id` al crear el modelo.

---

## Batería de casos de formato · 21-jul-2026

El corpus paraguayo es homogéneo: encabezados consistentes, sin numerales romanos,
ya anonimizado, con la recomendación siempre al final. Esa uniformidad permite que
el sistema sostenga supuestos sin que nada los cuestione.

Se construyó una batería de **16 casos de prueba sintéticos**
(`tests/casos_formato_chileno.py`) que reproducen variantes de redacción y
estructura documentadas en la práctica clínica local. Son informes ficticios:
nombres, identificadores y fechas inventados, ejecutables sin acceso a datos
clínicos.

### Qué cubre

| Grupo | Variantes |
|---|---|
| Escritura del BI-RADS | Arábigo, romano (`Birads -us III`), modalidad antepuesta (`US BIRADS 1`) y pospuesta (`Birads 2 US`), typo (`bi-radas`) |
| Estructura | `Impresión mamográfica` en vez de `Conclusión`, sin encabezado, descargo reubicado al inicio, mención histórica previa |
| Recomendación | Forma verbal (`controlar`), sinónimos de técnica (`ultrasonido`) |
| Comportamiento seguro | Hallazgos sin categoría, sospecha sin conducta, incoherencia crítica |
| Privacidad | Nombre e identificador pegados al texto clínico |

**Resultado: 16/16** en las cuatro dimensiones (categoría extraída, recomendación
clasificada, estado del cotejo, ausencia de identificadores tras la limpieza).

### Los numerales romanos

El corpus de entrenamiento contiene **0 %** de numerales romanos, pero la práctica
local sí los usa. El soporte se había programado como cobertura preventiva, sin
disponer de un ejemplo. Los casos `fmt_02` y `fmt_05` lo verifican.

Un modelo entrenado sobre el corpus no podría cubrir esa variante: nunca la vio.
Es un argumento concreto a favor de las reglas para una tarea donde la variación
es tipográfica y enumerable.

### Correcciones que motivó la batería

**1. Redacción intra-línea.** La limpieza operaba por líneas, así que un nombre de
radiólogo o un identificador en su propia línea se eliminaban, pero pegados a la
conclusión sobrevivían: borrar esa línea se habría llevado el BI-RADS, y el
fail-safe del 60 % lo impedía. La protección del contenido clínico anulaba la de
privacidad.

Solución: una segunda capa que **redacta dentro de la línea** en lugar de
borrarla, sustituyendo por `[MEDICO]`, `[RUT]` o `[DATO_PACIENTE]`. Se aplica
también cuando el fail-safe se dispara, que era el hueco real. Caso `priv_01`.

**2. El dashboard mostraba el texto crudo.** El panel de depuración y las vistas
previas enseñaban la entrada original, exponiendo en pantalla lo que la limpieza sí
redactaba. Corregido en las tres vistas: ahora muestran el texto ya procesado.

**3. Un subjuntivo confundido con un verbo gatillo.** La capa de tolerancia a typos
veía `sugieran` en *"imágenes que sugieran extravasación"* y lo tomaba por
`sugieren` (distancia de edición 1). Pero `sugieran` no es un typo: es español
válido con sentido descriptivo. Se añadió una lista de formas clínicas que nunca
deben tratarse como errores de tipeo.

**4. El título del examen leído como recomendación.** La regla de "directiva sin
verbo" tomaba `Ecotomografía Mamaria` (el título) y las referencias a estudios
previos como conductas a seguir. Dos guardas: se descarta el segmento si menciona
un año o un mes, y una directiva sin verbo solo se acepta en la segunda mitad del
informe, mismo criterio posicional que usa el buscador de BI-RADS.

**5. La forma verbal de "control".** *"Se sugiere controlar en seis meses"* no se
reconocía: todos los patrones exigían el sustantivo `control` seguido de espacio.
Ampliado a `control(?:ar|arse|arla|arlo)?`. Caso `rec_01`.

**6. La concordancia regex/NER mentía.** La comparación usaba contención de
subcadena, así que un span sobre-extraído que contenía dentro al correcto se
reportaba como "concuerdan". Eso ocultaba un fallo justo donde el contraste
debería revelarlo. Ahora exige equivalencia de largo (≥75 %) y añade el estado
`contenido_parcial`.

### Escalamiento de severidad por conducta ausente

El caso `seg_02` es un BI-RADS 5 con neoplasia y adenopatías de aspecto
metastásico que **no declara ninguna recomendación**. El sistema lo marcaba con
severidad *alta*, la misma que un BI-RADS 1 sin recomendación.

Se añadió el escalamiento: cuando el BI-RADS es 4, 5 o 6 y no hay recomendación
clasificable, la severidad pasa a **crítica** con la regla
`regla_sospecha_sin_recomendacion_declarada`. Es la contraparte de la alerta de
omisión del Módulo 1: allá falta la categoría, aquí falta la conducta.

### Verificación

Ninguna corrección produjo regresión: 8/8 tests del pipeline, 16/16 de la batería,
y 800 informes del corpus mantienen la misma distribución de estados. La capa de
redacción no genera falsos positivos sobre los 4 357 del corpus, que ya venía
anonimizado.

### Alcance

La batería verifica **cobertura de formatos**, no desempeño poblacional. Los casos
son sintéticos y no reemplazan la validación con un corpus clínico anotado del
contexto de despliegue, que sigue siendo la limitación de fondo.

---

## Lo que se probó y no quedó

**Embeddings como clasificador de la recomendación** (nb Matching). Se
definieron 38 anclas semánticas sobre 8 categorías y se asignó cada
recomendación a la categoría del ancla más cercana por similitud coseno.

| Modelo | Concordancia con reglas | Frases atípicas |
|---|---|---|
| Multilingüe genérico (1ª versión) | 69 % | 3/8 |
| **bsc-bio-ehr-es (clínico)** | **77,4 %** | **7/8** |
| sentence-es | 59,2 % | 8/8 |
| roberta-biomedical-es | 57,5 % | 8/8 |

Tras el 69 % inicial se diagnosticaron dos causas (vocabulario no clínico y
solapamiento entre las categorías de imagen) y se corrigieron ambas. El mejor
modelo clínico subió a 77,4 %. El umbral, fijado antes de correr el
experimento, era 90 %. Descartado con doble evidencia.

Causa: el 84 % de los errores es una sola confusión,
`estudio_complementario_imagen` clasificado como `correlacion_ecografica`
(828 de 982). Las tres categorías de imagen son semánticamente vecinas y un
clasificador por significado tiende a fundirlas. Las reglas las separan por
señales de intención explícitas.

**Embeddings como segundo revisor** (nb Segundo Revisor). El mismo modelo
ganador en un rol más modesto: correr en paralelo y marcar solo discrepancias
para revisión humana. Resultado mixto: 6/7 en frases atípicas (pasa), pero
26,97 % de discrepancias sobre el corpus (el umbral era 10 %). Un sistema que
marca uno de cada cuatro informes produce fatiga de alertas.

El 82,6 % de esas discrepancias es otra confusión única
(`correlacion_ecografica` como `control_anual`, 969 casos). Excluirla también
como ruido conocido habría bajado la tasa a 4,69 %, cerca de viable. No se
hizo: sería ajustar el experimento hasta obtener el resultado buscado.

**Predecir el BI-RADS desde los hallazgos** (nb Entrenamiento BI-RADS). Modelo
clínico en español, focal loss, pesos de clase inversos a la frecuencia, CV
5-fold estratificada, calibración por temperatura (T = 0,595), y eliminación de
tres fugas: el número declarado, la conclusión completa, y la recomendación.

Esta última fuga es la interesante. La recomendación es un proxy casi
determinista del BI-RADS: de los informes que recomiendan biopsia, 35 son
BI-RADS 4 y 14 son BI-RADS 5, y ninguno es 1, 2 o 3. Sin quitarla, el modelo
habría leído la recomendación en lugar de juzgar los hallazgos.

**Macro F1 out-of-fold = 0,6236.** El reporte por clase explica por qué no
alcanza:

| BI-RADS | F1 | Ejemplos |
|---|---|---|
| 0 | 0,96 | 949 |
| 1 | 0,98 | 435 |
| 2 | 0,98 | 2 326 |
| 3 | 0,72 | 87 |
| 4 | **0,31** | 52 |
| 5 | **0,41** | 16 |
| 6 | **0,00** | 5 |
| Exactitud | 0,95 | 3 870 |
| F1 ponderado | 0,95 | 3 870 |
| **Macro F1** | **0,62** | 3 870 |

Exactitud del 95 % y un modelo que nunca predice un cáncer confirmado. El
cuello de botella no es la técnica sino los datos: no se aprende a reconocer
algo que se ha visto cinco veces. La mejora estructural es recolectar más
ejemplos de las clases críticas con validación clínica.

---

## Estado actual

| Módulo | Método | Métrica |
|---|---|---|
| M0 · Limpieza | Reglas por línea, con salvaguarda | — |
| M1 · BI-RADS | Solo reglas, búsqueda híbrida en 4 fases | Macro F1 = 0,9995 |
| M2 · Recomendación | Reglas + NER DistilBETO de respaldo | Cobertura 99,82 % · NER F1 = 0,9991 · prueba de estrés: regla 0,5314 vs NER 0,9956 |
| M3 · Cotejo ACR | Tabla normativa auditable | 1,15 % de alertas (50 de 4 357) |

La IA quedó en el módulo donde la variación es semántica. El módulo donde la
variación es tipográfica quedó con reglas, porque los formatos de un número se
pueden enumerar y las redacciones de una recomendación clínica no.

## Pendiente

- Integrar la tolerancia a errores de tipeo (`proto_typos_birads.py`) como fase
  del buscador, o mantenerla como módulo separado.
- Validación clínica formal de la tabla ACR por radiólogos.
- Corpus chileno anotado. Es la limitación de fondo del trabajo. La batería de
  casos sintéticos verifica cobertura de formatos, no desempeño poblacional.
