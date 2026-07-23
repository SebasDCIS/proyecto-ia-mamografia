# Scripts auxiliares

Utilidades de diagnóstico y experimentación que no forman parte del pipeline,
pero que producen mediciones citadas en el informe.

| Script | Qué hace |
|---|---|
| `ablacion_leakage_birads.py` | Estudio de ablación del verificador DistilBETO. Enmascara la mención textual del número BI-RADS y reevalúa el modelo ya entrenado. Produce la caída de 0,939 a 0,544 que motivó retirar el Módulo 4. |
| `diagnostico_ml.py` | Diagnóstico del verificador DistilBETO: distribución de estados y comportamiento por nivel de confianza de la regla. Corresponde al módulo retirado; se conserva para reproducir su evaluación. |
| `crear_informes_prueba.py` | Genera informes de prueba en disco para validar el orquestador `predict.py` desde la línea de comandos. |

## Requisitos

Los dos primeros requieren el corpus descargado en `data/processed/` y el modelo
entrenado. Ver las instrucciones del README principal.

`crear_informes_prueba.py` no requiere nada externo.

## Nota

Para verificar el sistema sin descargar datos ni modelos, usar las suites del
README principal:

```bash
python -m src.predict
python -m tests.casos_formato_chileno
```
