# MEJORAS PENDIENTES — Sistema de Auditoría Mamográfica (BME513)

Estado a julio 2026. Proyecto: `github.com/SebasDCIS/proyecto-ia-mamografia`
Autor: Sebastián Inostroza Hurtado · Supervisor: Dr. Alejandro Veloz Baeza

> Registro vivo de deuda técnica, hallazgos metodológicos y decisiones
> pendientes. Ordenado por prioridad. Las prioridades marcadas 🔴 son
> bloqueantes para la defensa; 🟡 importantes; 🟢 mejoras.

---

## 0. Housekeeping inmediato

- [ ] **Commitear y pushear** los cambios de la sesión de validación (varios
      módulos parchados sin subir): `buscador_birads.py`, `extractor_recomendacion.py`,
      `extractor_birads.py`, `predict.py`, `dashboard/utils/formato.py`,
      `dashboard/pages/1_Informe_Individual.py`.
- [ ] Sugerencia de mensaje de commit: "fix(pipeline): cierra 4 bugs de
      validación real + panel de omisión en dashboard".

### Ya cerrado esta sesión (referencia, no pendiente)
- ✔ Detección de omisión de BI-RADS (Fase 4 de la Opción D) implementada y
  validada end-to-end; además independizada del verificador ML (funciona con `--no-ml`).
- ✔ Falso positivo por recomendación en prosa corregido (Vía 2.5, frases gatillo).
- ✔ Match espurio de "impresiona" como encabezado corregido (plurales + límite
  de palabra) + reconciliación de confianza vía buscador híbrido.
- ✔ Tests huecos T7/T8 reemplazados por aserciones reales.
- ✔ Panel de alerta de omisión en el dashboard (arreglaba además un `KeyError`
  que crasheaba la página con resultados de omisión).

---

## 1. 🔴 CRÍTICO — Fuga de etiqueta en DistilBETO

**Hallazgo (ablación por enmascaramiento, `ablacion_leakage_birads.py`):**
el modelo clasificador de documento explota la mención BI-RADS presente en el
texto de entrada.

- Macro F1 con el número visible: **0.9386**
- Macro F1 con el número enmascarado: **0.5438**
- Caída: **42%** (871/872 informes tenían la mención en el texto)
- Por clase: los benignos (BI-RADS 1 y 2) retienen señal real (F1 0.96 / 0.89);
  los sospechosos (0, 3, 4, 5) se desploman.

**Implicación:** el modelo *lee* el número, no lo *juzga* desde los hallazgos.
Esto invalida su rol original de "segunda opinión clínica independiente".
La métrica de CV (0.89) también está inflada por esta fuga.

**Acciones pendientes:**
- [ ] Decidir con Dr. Veloz el destino del modelo (ver §3).
- [ ] Si se mantiene con rol de juicio: **reentrenar enmascarando** la mención
      BI-RADS en la entrada (aplicar `enmascarar_birads()` a `X` antes de
      tokenizar, en notebooks 04 y 04b) y reportar la métrica honesta resultante.
- [ ] Actualizar informe LaTeX, dashboard y láminas: **no** presentar 0.89/0.94
      como medida de comprensión clínica. Distinguir explícitamente los dos tipos
      de fuga (ver nota abajo).

**Nota conceptual (para el informe y la defensa):** hay DOS tipos de fuga y hoy
solo uno está tratado.
- Fuga por **aumentación** (variantes del mismo informe en train y test):
  correctamente controlada (aumentación por fold en 04b). ✔
- Fuga de **etiqueta** (la respuesta está en el texto de entrada): detectada esta
  sesión, **no resuelta**. La afirmación "sin data leakage" actual solo cubre la
  primera; sharpen el discurso o se vuelve una vulnerabilidad en la defensa.

**Detalle colateral:** `max_length=256` trunca informes largos; si la conclusión
con el BI-RADS cae después del token 256, el modelo no la ve. Esto probablemente
explica por qué el F1 no fue ~1.0 pese a la fuga. Documentarlo.

---

## 2. ✅ Detección de negación (extensión Fase 2 del buscador) — IMPLEMENTADA

**Estado: cerrada.** Integrada como regla en el filtrado contextual de la Fase 2,
junto a histórico/comparativo/educacional.

- `es_mencion_negada()` con patrones regex estilo NegEx (toleran palabras
  intermedias: "no **son** sugerentes de").
- Criterio conservador (alta precisión): solo niega aserciones de categoría
  ("no corresponde a", "no sugerente de", "sin criterios de", "se descarta que
  corresponda"). NO se dispara con "no se observa nódulo" ni "descartar patología".
- Tests inline agregados (T6 negación, T7 anti-falso-positivo). Buscador 7/7.

**Pendiente menor (opcional):** evaluar migrar a `negspacy` (NegEx validado) si
aparecen negaciones más complejas en informes reales.

---

## 3. ✅ Rol del componente ML — DECIDIDO

**Decisión tomada:** el ML queda como **apoyo de lectura del BI-RADS** (opción a+b
parcial). Reencuadre aplicado en código, dashboard e informe de defensa.

- **(a) Árbitro de empates** (Fase 4, implementado): elige entre menciones
  candidatas empatadas por posición. Tarea de *lectura* → dentro de su competencia.
- **Apoyo de lectura** (implementado): relee la mención en su ventana local y
  refuerza/cuestiona la extracción literal, que es la autoridad.
- **(c) Lector de respaldo por confianza baja: DESCARTADO.** Opera justo donde el
  modelo lee peor (informes difíciles), y esa función ya la cubre mejor y de forma
  auditable el buscador híbrido. Descartado por criterio; opcionalmente medible
  con el script de recuperación (§5) si se quiere respaldar con un número.
- **(d) Retirar el clasificador: no.** Se mantiene con rol acotado de apoyo.

Nota: (a) y apoyo de lectura NO sufren fuga —el trabajo es leer/elegir números
presentes, no predecirlos—.

---

## 4. 🔴 Reconciliar inconsistencias antes de defender

## 4. ✅ Reconciliación de cifras — CERRADA (con el dataset)

Confirmadas directamente sobre `reports_cleaned.csv`:

- [x] **Corpus:** 4.357 informes (todos con BI-RADS), 4.347 con recomendación
      (10 sin). No era inconsistencia: son dos subconjuntos.
- [x] **Desbalance de clases documentado:** BI-RADS 2 = 60%; 4/5/6 = 52/16/5
      ejemplos. Justifica Macro F1 y refuerza el reencuadre del ML.
- [x] **Tasa de alertas:** 1.33% (58 alertas), medida con el pipeline actual
      sobre el corpus completo. 2 críticas, 38 altas, 17 medias, 1 baja.
      **0 omisiones** (correcto: el corpus tiene BI-RADS en todos) → detector de
      omisión validado con ~0 falsos positivos. La cifra 1.0%/44 de tabla_acr era
      previa; el dashboard ya muestra 1.33%.
- [x] Dashboard reencuadrado con cifras medidas: exactitud extracción 99.9%,
      apoyo ML ≈0.89 (CV, lectura), tasa 1.33%.
- [x] **Métrica oficial DECIDIDA** (sin disponibilidad de Dr. Veloz): se reporta
      exactitud de extracción 99.9% (medida vs. verdad) como cifra principal del
      sistema, y Macro F1 CV ≈0.89 como *lectura* del apoyo ML, con el contexto de
      la ablación. Punto de honestidad explícito: el ML no mejora la extracción
      (la regex ya lee mejor); es un cross-check independiente. Documentado en §6.1
      y §7.5 del informe de defensa.
- [ ] Pendiente (tú): propagar estas cifras al informe LaTeX y las láminas;
      confirmar país/origen del corpus.

### Bugs encontrados y corregidos en la validación de corpus (esta sesión)

- [x] **60 falsos positivos de omisión** por sobre-descarte de la Fase 2
      ("anterior"/"vs." en la ventana sin modificar el BI-RADS). Fix: cuando se
      descartan TODAS las menciones, replegarse a ranking posicional (confianza
      baja) en vez de declarar omisión. La omisión real solo si Fase 1 vacía.
- [x] **Typo "BI-RADS O"** (letra o por cero) que la Fase 1 del buscador no
      capturaba. Fix: cross-check en `predict.py` — solo declarar omisión si el
      extractor regex TAMPOCO encontró BI-RADS.
- [ ] Pendiente (opcional): mejorar la *precisión* de los filtros histórico/
      comparativo de la Fase 2 (que matcheen por adyacencia, no por presencia en
      ventana). Hoy mitigado por el repliegue; documentado como limitación.

---

## 5. 🟢 Experimentos de validación que fortalecen la defensa

- [ ] **Recuperación en regex-baja:** en el subconjunto donde la regex quedó en
      confianza baja/no_detectado, medir cuántas veces el modelo (o el buscador)
      recupera el BI-RADS declarado correcto. Justifica con números la función de
      "lector de respaldo".
- [ ] **Extracción sobre informes difíciles:** aislar ~30-50 informes con
      menciones múltiples, históricas, negaciones o sin encabezado; con su
      BI-RADS declarado correcto (ground truth), comparar regex sola vs. buscador
      híbrido. Es la evidencia que justifica el módulo bajo el objetivo real
      (extracción robusta, no predicción).
- [ ] **Validación amplia del detector de omisión:** los vocabularios
      (`HALLAZGOS_DESCRIPTORES`, `ACCIONES_CLINICAS`) se calibraron con pocos
      ejemplos. Revisar falsos positivos/negativos sobre un conjunto amplio.

---

## 6. 🟢 Despliegue (demo compartible)

Analizado esta sesión. Prioridad baja (opcional).

- [ ] Si se quiere link estable: **Hugging Face Spaces** (16 GB RAM gratis,
      soporta el modelo). Streamlit Community Cloud queda chico (1 GB) salvo modo
      `--no-ml` o aumento educativo.
- [ ] Si es para mostrar en vivo: **túnel** (`cloudflared`/`ngrok`) desde el Mac
      — mantiene el procesamiento local (mejor para privacidad).
- [ ] **Requisito de privacidad:** cualquier despliegue cloud SOLO con informes
      sintéticos/anonimizados; ponerlo explícito en la app. Nunca datos de
      pacientes reales sin resguardos (Ley 19.628 / 21.719).
- [ ] Resolver hosting del modelo (git-LFS o repo de modelo en HF) y mover el
      `config.toml` a la raíz si se usa Streamlit Cloud.

---

## 7. 🟢 Mejoras técnicas menores

- [ ] **Reconciliación de confianza** cuando el extractor devuelve `None` pero el
      buscador sí encuentra categoría (hoy solo reconcilia cuando ambos coinciden
      en valor con confianza baja).
- [ ] Revisar si `birads.fuente = "buscador_hibrido_posicional"` (nuevo string en
      informes sin encabezado) rompe algún consumidor del JSON o del dashboard.
- [ ] El warning cosmético de `urllib3 / LibreSSL` en macOS: inofensivo, se puede
      silenciar si molesta.
- [ ] Páginas planificadas del dashboard: Procesamiento Batch y Estadísticas del
      Corpus (marcadas "próximamente").

---

## 8. Estado de documentación

- [x] Documento de defensa (`defensa_proyecto_BME513.md`) creado.
- [ ] Actualizar §7-8 del documento de defensa con la fuga de etiqueta y el
      número real tras reentrenar (o tras decidir el rol del ML).
- [ ] Actualizar `RESUMEN_PROYECTO.md` con el estado post-validación.
- [ ] Documentar en notebooks el hallazgo de fuga y la ablación (trazabilidad).

---

_Última actualización: sesión de validación sobre informes reales, julio 2026._
