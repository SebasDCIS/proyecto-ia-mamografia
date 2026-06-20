# Mejoras Pendientes — Proyecto BME513

Documento de seguimiento para próximas sesiones de trabajo.

## Prioridad ALTA

### 1. Búsqueda semántica para verificador ML (Problema 1)

**Problema actual**: El verificador ML solo busca en el bloque CONCLUSIÓN.
Si el informe no tiene encabezado de conclusión bien marcado, el ML falla.

**Solución propuesta — Opción 3: Búsqueda híbrida regex + ranking ML**

1. Regex amplia: encuentra TODAS las menciones de BI-RADS en el informe
2. ML rankea cuál mención es "la conclusión final" vs referencia histórica
3. Regex extrae el número de la mención mejor rankeada

**Beneficios**:
- Regex sigue siendo autoridad clínica (interpretable)
- ML solo ayuda a localizar, no a razonar
- Cero solapamiento con coherence-audit
- Maneja informes con estructura no estándar

**Tecnologías**:
- sentence-transformers para embeddings semánticos en español
- Modelo: paraphrase-multilingual-MiniLM-L12-v2 o BETO embeddings

**Estimación**: 1 sesión completa (4-5 horas)

---

### 2. Selector manual de recomendación en dashboard (Problema 2)

**Problema actual**: Cuando el extractor de recomendaciones no clasifica el
texto, el sistema no puede ejecutar el cotejo.

**Solución propuesta — A + E combinadas**:

A. Expandir el vocabulario regex con más sinónimos clínicos
E. Cuando aún así no clasifica, mostrar dropdown con las 8 categorías para
   que el usuario elija manualmente:
   - biopsia_histologia
   - derivacion_oncologica
   - estudio_complementario_imagen
   - correlacion_ecografica
   - comparacion_estudios_previos
   - control_corto_plazo
   - control_anual
   - criterio_medico

**Estimación**: 2-3 horas

---

## Prioridad MEDIA

### 3. Página 2 del dashboard — Procesamiento Batch

- Upload de carpeta completa con múltiples informes (TXT/PDF)
- Procesamiento con barra de progreso
- Tabla de resultados ordenados por severidad
- Filtros: solo alertas, por BI-RADS, por confiabilidad
- Exportar resultados a CSV

**Estimación**: 1 sesión (3-4 horas)

---

### 4. Página 3 del dashboard — Estadísticas del Corpus

- Visualización de los resultados ya generados (notebook 09)
- Distribución de BI-RADS, estados de cotejo
- Matriz de confusión DistilBETO
- Casos críticos del corpus

**Estimación**: 1 sesión (2-3 horas)

---

## Prioridad BAJA

### 5. Completar informe v8 (LaTeX)

- Sección III: Resultados con tablas y figuras
- Sección IV: Conclusiones y trabajo futuro
- Compilar PDF final
- Revisión y submission al curso BME513

**Estimación**: 1-2 sesiones

---

### 6. Validación clínica formal

- Identificar 2-3 radiólogos chilenos colaboradores
- Validación cruzada de las alertas detectadas
- Ajuste de la tabla normativa con feedback local

**Estimación**: Externa, depende de disponibilidad de radiólogos

---

## Próxima sesión sugerida

**Empezar por la mejora 1** (búsqueda semántica para módulo 4).

Es la más impactante metodológicamente, y resuelve directamente el problema
de los informes que fallan por falta de bloque CONCLUSIÓN claro.

Una vez implementada, re-evaluamos el sistema sobre el corpus completo.

---

## Estado al cierre de esta sesión

- Notebook 09: integración cotejo + verificador ML
- Cotejo v2: integración exitosa, 100/100 consistencia con orquestador
- Notebook 10: orquestador documentado
- src/predict.py: orquestador con CLI y soporte TXT/PDF
- Dashboard Streamlit: landing + página 1 (informe individual)
- Informe v8: borrador con abstract, intro y métodos

Pendiente:
- Secciones III y IV del informe v8
- Mejoras 1 y 2 (búsqueda semántica + selector manual)
- Páginas 2 y 3 del dashboard
