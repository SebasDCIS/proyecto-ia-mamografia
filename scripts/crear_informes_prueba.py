#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera los informes de prueba para validar el orquestador predict.py.

- informe_real.txt    : caso COHERENTE (BI-RADS asignado correctamente).
                        Sin encabezado de sección -> prueba el buscador híbrido
                        con scoring posicional (el método antiguo fallaría aquí).
- informe_omision.txt : caso de OMISIÓN (hallazgos sospechosos + recomendación
                        de biopsia, pero SIN categoría BI-RADS asignada).

Ejecutar desde la raíz del repo:
    python crear_informes_prueba.py
"""

# --------------------------------------------------------------------------
# INFORME 1 -> caso coherente (real). BI-RADS presente al final, sin encabezado.
# --------------------------------------------------------------------------
informe_real = """Examen realizado como estudio complementario a RM.
Se explora dirigidamente mama izquierda.
Se visualiza conducto central prominente en región retroareolar, sin objetivarse nódulos en su interior. En región periareolar de cuadrante inferoexterno, en radio de la hora 5 se observa conducto prominente y tortuoso, con contenido ecogénico en su interior, conformando una imagen de aproximadamente 18 mm, que impresiona en concordancia con el área de realce observada en RM. No presenta signos de hipervascularización al estudio con Doppler power.
Se sugiere realizar estudio histológico de esta imagen para estudio de probables lesiones papilares, idealmente con biopsia radio quirúrgica.
Birads 4.
"""

# --------------------------------------------------------------------------
# INFORME 2 (sin la línea "Birads 4.") -> caso de OMISIÓN.
# Tiene "Impresión Diagnóstica:" y menciones a estudios históricos,
# pero NO asigna categoría BI-RADS actual.
# --------------------------------------------------------------------------
informe_omision = """Paciente en estudio por telorraquia izquierda, que la paciente refiere de aprox. 5 años de evolución.
Estudio citológico realizado en mama izquierda habría demostrado papiloma.
Además en control por microcalcificaciones mamarias derechas.
Se dispone de mamografía bilateral de febrero 2023, y mamografía unilateral derecha reciente. Se realiza en forma complementaria ecotomografía de second look.
Regular cantidad de tejido fibroglandular en ambas mamas, con patrón de realce de fondo escaso, simétrico.
En mama izquierda se observa leve dilatación ductal central con hiperseñal en secuencias T1. Se reconoce un pequeño foco de realce nodular de 3mm inmediatamente superior al árbol ductal, que realza significativamente con el contraste.
En unión de cuadrantes inferiores, e inmediatamente inferior y conectado al árbol ductal dilatado se observa área de hiperrealce de tipo no masa de distribución segmentaria, de aprox. 26mm de diámetro mayor. En su borde lateral se observa pequeño foco de realce nodular de 4mm.
Se realiza ecografía de second look en la que se observa conducto central con leve ectasia, y conducto en CIE periareolar con contenido ecogénico, que impresiona en correspondencia con área de hiperrealce (imágenes en estudio de ecografía complementaria).
Imágenes quísticas aisladas en ambas mamas, la mayor en CII de mama derecha en concordancia con nódulo mamográfico.
No se observan otros nódulos sospechosos ni otras áreas de realce anormales con el contraste. No se observan alteraciones en área de microcalcificaciones mamográficas derechas.
No se observan adenopatías sospechosas.
No se observa realce anormal de estructuras musculares de la pared torácica ni del plano cutáneo.
Complejos aréola-pezón de aspecto conservado.
Impresión Diagnóstica:
Leve ectasia ductal retroareolar central de mama izquierda, con pequeño nódulo en localización superior y área de realce segmentario en unión cuadrantes inferiores, que pueden corresponder a lesiones papilares.
Se sugiere estudio histológico idealmente por biopsia radioquirúrgica, o en su defecto bajo guía ecográfica dirigido al área de realce en UCInf.
"""

archivos = {
    "informe_real.txt": informe_real,
    "informe_omision.txt": informe_omision,
}

for nombre, contenido in archivos.items():
    with open(nombre, "w", encoding="utf-8") as f:
        f.write(contenido)
    n_lineas = contenido.count("\n")
    tiene_birads = "birads" in contenido.lower()
    print(f"[OK] {nombre:22s} ({n_lineas} lineas) | BI-RADS presente: {tiene_birads}")

print("\nListo. Ahora puedes ejecutar:")
print("  python -m src.predict --input informe_real.txt    --id caso_real     -o resultado_real.json")
print("  python -m src.predict --input informe_omision.txt --id caso_omision  -o resultado_omision.json")
