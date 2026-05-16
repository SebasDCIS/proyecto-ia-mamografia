# Anexos del proyecto

Esta carpeta contiene las salidas generadas por los notebooks de la fase experimental.
Sirven como evidencia reproducible de los resultados reportados en el informe parcial.

## Matrices de confusión normalizadas

- `confmat_linearsvc_balanced.png` — LinearSVC balanced (nb 02)
- `confmat_distilbert_eng_plus_aug.png` — DistilBERT inglés + augmentación (nb 03)
- `confmat_distilbeto_plus_aug.png` — DistilBETO + augmentación (nb 04, modelo final)

## Classification reports por clase

- `report_linearsvc_balanced.csv` — métricas por BI-RADS para LinearSVC balanced
- `report_distilbert_eng_plus_aug.csv` — métricas por BI-RADS para DistilBERT inglés + aug
- `report_distilbeto_plus_aug.csv` — métricas por BI-RADS para DistilBETO + aug

## Predicciones crudas

- `predicciones_distilbeto.json` — y_test e y_pred del modelo final, útiles para
  regenerar la matriz de confusión y otras métricas sin reentrenar.

## Resumen de la validación cruzada (DistilBETO)

Generado por el notebook `04b_cv_distilbeto.ipynb`:
`../results_distilbeto_cv/resumen_cv.json` (excluido del repo por tamaño).
