# Informe LaTeX

Código fuente del informe del proyecto BME513.

## Archivos

```
Informe_BME513_v8.tex     Fuente principal
referencias.bib           Bibliografía (BibTeX)
figuras/                  Figuras en PDF vectorial
Informe_BME513_v8.pdf     PDF compilado
Informe_BME513_v8.bbl     Bibliografía procesada (se versiona a propósito, ver abajo)
```

## Compilación

El documento usa la clase `IEEEtran` y BibTeX. La secuencia completa es:

```bash
pdflatex Informe_BME513_v8
bibtex   Informe_BME513_v8
pdflatex Informe_BME513_v8
pdflatex Informe_BME513_v8
```

Los cuatro pasos son necesarios y en ese orden. Con menos, las citas quedan sin
resolver y aparecen como `[?]`.

Editores como Overleaf, TeXShop o TeXstudio ejecutan la secuencia
automáticamente; desde terminal hay que correrla a mano.

Alternativa en un solo comando, si está disponible `latexmk`:

```bash
latexmk -pdf Informe_BME513_v8.tex
```

## Sobre el archivo .bbl

El `.bbl` es un archivo generado, y lo habitual es no versionarlo. Aquí se
incluye a propósito: permite recompilar el PDF con una sola pasada de `pdflatex`
y sin ejecutar BibTeX, lo que evita que las citas salgan como `[?]` en una
compilación rápida.

## Paquetes requeridos

`IEEEtran`, `graphicx`, `booktabs`, `amsmath`, `cite`, `hyperref`, `xcolor`,
`array` y `babel` con soporte de español. Todos vienen en una instalación
completa de TeX Live o MacTeX.
