# Dashboard de Auditoría Mamográfica

Interfaz web del sistema de auditoría técnica de informes mamográficos
del proyecto BME513.

## Cómo ejecutarlo

Desde la raíz del proyecto (`proyecto-ia-mamografia/`):

```bash
streamlit run dashboard/app.py
```

El dashboard se abre automáticamente en el navegador en `http://localhost:8501`.

Para detenerlo: Ctrl+C en la terminal.

## Páginas disponibles

### 📝 Informe Individual

Procesa **un** informe mamográfico a la vez con tres modos de entrada:

- **Pegar texto:** copia y pega el informe completo en un textarea
- **Archivo TXT:** sube un archivo `.txt` con el informe
- **Archivo PDF:** sube un PDF con texto digital (no escaneado)

Muestra el resultado con:
- Banner de estado tipo semáforo (verde / amarillo / naranja / rojo)
- Resumen del caso (BI-RADS extraído, recomendación esperada vs detectada)
- Verificación dual (regex + DistilBETO)
- Mensaje clínico explicativo
- JSON completo descargable

### 📁 Procesamiento Batch (próximamente)

Procesar múltiples informes en una sola operación. Genera una tabla
con todas las alertas detectadas ordenadas por severidad.

### 📊 Estadísticas del Corpus (próximamente)

Visualizaciones agregadas sobre el corpus validado.

## Estructura del dashboard

```
dashboard/
├── app.py                          # Landing page
├── pages/
│   └── 1_Informe_Individual.py     # Página 1
├── utils/
│   ├── __init__.py
│   └── formato.py                  # Helpers de visualización
├── .streamlit/
│   └── config.toml                 # Configuración (telemetría off)
└── README.md
```

## Privacidad

- La telemetría anónima de Streamlit está **desactivada** en `.streamlit/config.toml`
- Todo el procesamiento es **local**, no se envían datos a servicios externos
- Cumple con la Ley N.º 19.628 sobre protección de datos personales (Chile)

## Tecnologías

- **Streamlit 1.50+** para la interfaz
- **pdfplumber 0.11+** para extracción de texto desde PDF
- Backend: el módulo `src/predict.py` (pipeline completo del proyecto)
