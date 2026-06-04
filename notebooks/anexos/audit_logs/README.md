# Audit Logs

Esta carpeta contiene los reportes JSON de auditoria generados por
`src/cotejo_acr.py` al procesar el corpus desde el notebook 07.

## Contenido

Cada archivo JSON corresponde a una alerta del sistema (BI-RADS vs
recomendacion incoherente) y contiene:
- Texto original y normalizado de la recomendacion
- Categorias clinicas detectadas y categoria principal
- Regla ACR aplicada y severidad
- Trazabilidad completa para validacion humana

## Regeneracion

Los archivos se regeneran ejecutando el notebook 07 completo. No se
versionan en Git porque son outputs derivados que se reproducen de
forma determinista a partir del corpus + modulos del MVP.

Solo este README esta versionado para preservar la estructura del
directorio.
