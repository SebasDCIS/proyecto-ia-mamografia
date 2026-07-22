"""
Batería de casos de prueba sintéticos: variantes de formato de informes
mamográficos en la práctica clínica chilena.

Estos informes son FICTICIOS. Nombres, identificadores y fechas son inventados;
no corresponden a personas ni a instituciones reales. Su función es cubrir
variantes de redacción y estructura que el corpus de entrenamiento (paraguayo,
Vázquez Noguera et al. 2025) no contiene, y verificar que el sistema las maneja.

Cada caso declara el resultado esperado, de modo que la batería sirve como
prueba de regresión reproducible: cualquiera puede ejecutarla sin acceso a
datos clínicos.

Uso:
    python -m tests.casos_formato_chileno
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Encabezado y pie ficticios, comunes a varios casos.
# Reproducen la estructura administrativa habitual de un informe de clínica
# privada: datos del paciente arriba, firma y descargo legal abajo.
# ---------------------------------------------------------------------------

_DESCARGO = (
    "*El presente resultado debe correlacionarse con el cuadro clinico "
    "y ser evaluado por medico tratante*"
)


def _encabezado(nombre: str, edad: str, rut: str, fecha: str) -> str:
    return (
        f"Nombre Paciente : {nombre}\n"
        f"Edad : {edad}\n"
        f"RUT : {rut}\n"
        f"Fecha Examen : {fecha}\n"
    )


def _firma(medico: str) -> str:
    return (
        f"Dr. {medico}\n"
        f"Médico Radiólogo\n"
        f"Informe validado por: /Dr(a). {medico}\n"
    )


# ---------------------------------------------------------------------------
# CASOS
#
# 'espera' declara lo que el sistema debe producir. Los campos con valor None
# indican que no debe extraerse nada en ese punto.
# ---------------------------------------------------------------------------

CASOS: List[Dict[str, Any]] = [

    # -----------------------------------------------------------------------
    # Grupo 1: variantes de escritura de la categoría BI-RADS
    # -----------------------------------------------------------------------
    {
        "id": "fmt_01_arabigo_estandar",
        "variante": "Formato canónico con encabezado Conclusión",
        "texto": (
            _encabezado("Perez Soto, Ana Maria", "52A 3M", "11111111-1", "01-01-2026") +
            "Mamografía Bilateral\n"
            "Antecedentes clínicos:\n"
            "Control de rutina.\n\n"
            "Hallazgos:\n"
            "Parénquima mamario heterogéneo y con densidad normal.\n"
            "No se identifican nódulos ni microcalcificaciones agrupadas sospechosas.\n\n"
            "Conclusión:\n"
            "Examen sin hallazgos sospechosos.\n"
            "Birads 2\n"
            "ACR b\n"
            "Se sugiere control anual\n\n" +
            _firma("Juan Ignacio Rojas Vera") + _DESCARGO
        ),
        "espera": {"birads": 2, "categoria": "control_anual", "estado": "coherente"},
    },
    {
        "id": "fmt_02_romano_con_modalidad",
        "variante": "Numeral romano con etiqueta de modalidad (ausente del corpus)",
        "texto": (
            _encabezado("Gomez Lira, Carmen Rosa", "55A 1M", "22222222-2", "02-01-2026") +
            "ECOTOMOGRAFÍA MAMARIA\n"
            "Antecedentes clínicos:\n"
            "Consulta por control. Se dispone de mamografía simultánea.\n\n"
            "Hallazgos:\n"
            "Tejido fibroglandular abundante y heterogéneo.\n"
            "Se observa un quiste tabicado en el cuadrante supero externo derecho, de 6 mm.\n\n"
            "Impresión diagnóstica:\n"
            "Quiste tabicado derecho que se sugiere controlar en seis meses.\n"
            "Birads -us III\n\n" +
            _firma("Marta Elena Vidal Contreras") + _DESCARGO
        ),
        "espera": {"birads": 3, "categoria": "control_corto_plazo", "estado": "coherente"},
    },
    {
        "id": "fmt_03_modalidad_prefijo",
        "variante": "Modalidad antepuesta: 'US BIRADS 1'",
        "texto": (
            _encabezado("Silva Nunez, Patricia Elena", "34A 8M", "33333333-3", "03-01-2026") +
            "Ecotomografía Mamaria\n"
            "Paciente referida para control.\n"
            "Patrón mamario mixto.\n"
            "No se reconocen nódulos sólidos mamarios.\n\n"
            "Impresión diagnóstica:\n"
            "Examen sin signos sospechosos de malignidad.\n"
            "US BIRADS 1.\n"
            "Se sugiere control anual.\n\n" +
            _firma("Pedro Antonio Lagos Miranda") + _DESCARGO
        ),
        "espera": {"birads": 1, "categoria": "control_anual", "estado": "coherente"},
    },
    {
        "id": "fmt_04_modalidad_sufijo",
        "variante": "Modalidad pospuesta: 'Birads 2 US'",
        "texto": (
            _encabezado("Castro Fuentes, Elena Paz", "66A 2M", "44444444-4", "04-01-2026") +
            "Ecotomografía Mamaria\n"
            "Antecedentes de neoplasia mamaria derecha tratada.\n"
            "Examen de control.\n\n"
            "Hallazgos:\n"
            "Cicatriz quirúrgica en cuadrantes externos de mama derecha.\n"
            "No se observan lesiones nodulares sólidas o áreas sospechosas.\n\n"
            "Impresión diagnóstica:\n"
            "Examen sin evidencias de lesiones sospechosas.\n"
            "Birads 2 US.\n"
            "Se sugiere control anual con mamografía y ecografía.\n\n" +
            _firma("Sofia Isabel Herrera Pinto") + _DESCARGO
        ),
        "espera": {"birads": 2, "categoria": "control_anual", "estado": "coherente"},
    },
    {
        "id": "fmt_05_guion_minuscula",
        "variante": "Guion y minúscula: 'Birads -us II'",
        "texto": (
            _encabezado("Morales Diaz, Isabel Cristina", "58A 5M", "55555555-5", "05-01-2026") +
            "ECOTOMOGRAFÍA MAMARIA\n"
            "Antecedentes clínicos:\n"
            "Consulta por control.\n\n"
            "Hallazgos:\n"
            "Pequeño quiste simple y aislado de 6 mm en mama derecha.\n"
            "No se observan adenopatías patológicas axilares.\n\n"
            "Impresión diagnóstica:\n"
            "Examen sin hallazgos sospechosos\n"
            "Birads -us II\n"
            "Se sugiere control anual\n\n" +
            _firma("Ricardo Andres Munoz Tapia") + _DESCARGO
        ),
        "espera": {"birads": 2, "categoria": "control_anual", "estado": "coherente"},
    },
    {
        "id": "fmt_06_error_tipeo",
        "variante": "Error de tipeo en el token: 'bi-radas'",
        "texto": (
            _encabezado("Vega Cortes, Lorena Andrea", "47A 9M", "66666666-6", "06-01-2026") +
            "Mamografía Bilateral\n"
            "Hallazgos:\n"
            "Se observa nódulo sólido irregular en cuadrante superior externo izquierdo.\n\n"
            "Impresión mamográfica.\n"
            "Nódulo de aspecto sospechoso.\n"
            "bi-radas 4.\n"
            "Se sugiere estudio histológico mediante biopsia.\n\n" +
            _firma("Claudio Esteban Salas Bravo") + _DESCARGO
        ),
        "espera": {"birads": 4, "categoria": "biopsia_histologia", "estado": "coherente"},
    },

    # -----------------------------------------------------------------------
    # Grupo 2: variantes estructurales
    # -----------------------------------------------------------------------
    {
        "id": "est_01_impresion_diagnostica",
        "variante": "Encabezado 'Impresión mamográfica' en lugar de 'Conclusión'",
        "texto": (
            _encabezado("Torres Bravo, Marcela Ines", "50A 6M", "77777777-7", "07-01-2026") +
            "Mamografía Bilateral\n"
            "Referida para control preventivo.\n"
            "Ambas mamas presentan patrón de densidades fibroglandulares dispersas.\n"
            "Persiste un nódulo de baja densidad en región periareolar izquierda, sin progresión.\n\n"
            "Impresión mamográfica.\n"
            "Nódulo mamario izquierdo, estable, de aspecto benigno.\n"
            "Birads 2.\n"
            "ACR B.\n"
            "Se sugiere control mamográfico anual.\n\n" +
            _firma("Andrea Paz Figueroa Leon") + _DESCARGO
        ),
        "espera": {"birads": 2, "categoria": "control_anual", "estado": "coherente"},
    },
    {
        "id": "est_02_sin_encabezado",
        "variante": "Sin encabezado de conclusión: la categoría va en prosa",
        "texto": (
            _encabezado("Rios Pena, Veronica Alejandra", "44A 1M", "88888888-8", "08-01-2026") +
            "Ecotomografía Mamaria\n"
            "Paciente referida para evaluación por mamas densas.\n"
            "Patrón fibroglandular homogéneo en ambas mamas.\n"
            "Grupo de pequeños quistes en CSE de mama derecha de 7 mm.\n"
            "No se observan otros nódulos sólidos ni quísticos.\n"
            "Examen sin signos sospechosos de malignidad. Birads 2.\n"
            "Se sugiere control anual con mamografía y ecografía.\n\n" +
            _firma("Gonzalo Javier Espinoza Ruiz") + _DESCARGO
        ),
        "espera": {"birads": 2, "categoria": "control_anual", "estado": "coherente"},
    },
    {
        "id": "est_03_descargo_reubicado",
        "variante": "Descargo legal al inicio (reordenamiento al extraer de PDF)",
        "texto": (
            _DESCARGO + "\n" +
            _encabezado("Nunez Salas, Daniela Paz", "61A 4M", "99999999-9", "09-01-2026") +
            "Mamografía Bilateral\n"
            "Hallazgos:\n"
            "Mamas constituidas por opacidades fibroglandulares dispersas.\n"
            "No se identifican imágenes nodulares dominantes.\n\n"
            "Impresión diagnóstica:\n"
            "Mamografía sin hallazgos sugerentes de malignidad.\n"
            "Birads 2. ACR B.\n"
            "Se sugiere control anual con mamografía.\n\n" +
            _firma("Fernanda Loreto Castillo Aravena")
        ),
        "espera": {"birads": 2, "categoria": "control_anual", "estado": "coherente"},
    },
    {
        "id": "est_04_mencion_historica",
        "variante": "Mención histórica antes de la definitiva",
        "texto": (
            _encabezado("Guzman Rojas, Claudia Beatriz", "57A 7M", "10101010-1", "10-01-2026") +
            "Mamografía Bilateral\n"
            "Antecedente: mamografía de 2023 informada como BI-RADS 2.\n"
            "Se observan calcificaciones agrupadas de aspecto pleomorfo en cuadrante "
            "superior externo derecho.\n\n"
            "Impresión diagnóstica:\n"
            "Calcificaciones de aspecto sospechoso.\n"
            "Birads 4.\n"
            "Se sugiere biopsia estereotáxica.\n\n" +
            _firma("Matias Alberto Fuentes Reyes") + _DESCARGO
        ),
        "espera": {"birads": 4, "categoria": "biopsia_histologia", "estado": "coherente"},
    },

    # -----------------------------------------------------------------------
    # Grupo 3: redacción de la recomendación
    # -----------------------------------------------------------------------
    {
        "id": "rec_01_forma_verbal",
        "variante": "Forma verbal 'controlar' en lugar del sustantivo 'control'",
        "texto": (
            _encabezado("Aguilar Mena, Ximena Loreto", "49A 2M", "12121212-2", "11-01-2026") +
            "Ecotomografía Mamaria\n"
            "Hallazgos:\n"
            "Nódulo sólido hipoecogénico de 8 mm en mama derecha, sin cambios.\n\n"
            "Impresión diagnóstica:\n"
            "Nódulo de aspecto benigno que se sugiere controlar en seis meses.\n"
            "BI-RADS 3.\n\n" +
            _firma("Cristian Eduardo Nunez Paredes") + _DESCARGO
        ),
        "espera": {"birads": 3, "categoria": "control_corto_plazo", "estado": "coherente"},
    },
    {
        "id": "rec_02_sinonimo_tecnica",
        "variante": "Sinónimo de la técnica: 'ultrasonido' por 'ecografía'",
        "texto": (
            _encabezado("Sepulveda Ortiz, Javiera Ines", "41A 11M", "13131313-3", "12-01-2026") +
            "Mamografía Bilateral\n"
            "Hallazgos:\n"
            "Asimetría focal en mama izquierda que requiere caracterización.\n\n"
            "Conclusión:\n"
            "Hallazgo que amerita estudio complementario.\n"
            "Birads 0.\n"
            "Se sugiere complementar con ultrasonido mamario.\n\n" +
            _firma("Valentina Andrea Soto Miranda") + _DESCARGO
        ),
        "espera": {"birads": 0, "categoria": None, "estado": None},
    },

    # -----------------------------------------------------------------------
    # Grupo 4: comportamiento seguro ante información faltante
    # -----------------------------------------------------------------------
    {
        "id": "seg_01_hallazgos_sin_birads",
        "variante": "Hallazgos y recomendación, pero sin categoría BI-RADS",
        "texto": (
            _encabezado("Contreras Vera, Antonia Paz", "53A 3M", "14141414-4", "13-01-2026") +
            "Mamografía Bilateral\n"
            "Hallazgos:\n"
            "Se observa nódulo espiculado de 12 mm en cuadrante superior externo izquierdo, "
            "de bordes irregulares.\n\n"
            "Impresión diagnóstica:\n"
            "Nódulo de aspecto sospechoso.\n"
            "Se sugiere biopsia.\n\n" +
            _firma("Rodrigo Ignacio Vergara Campos") + _DESCARGO
        ),
        "espera": {"birads": None, "categoria": None, "estado": "no_procesable"},
    },
    {
        "id": "seg_02_sospecha_sin_conducta",
        "variante": "BI-RADS de sospecha sin recomendación declarada",
        "texto": (
            _encabezado("Herrera Lopez, Bernardita Elena", "68A 5M", "15151515-5", "14-01-2026") +
            "Ecotomografía Mamaria\n"
            "Antecedentes clínicos\n"
            "Masa palpable en mama derecha.\n\n"
            "Hallazgos\n"
            "Se observa nódulo sólido irregular, heterogéneo e hipoecogénico, de bordes "
            "espiculados, que compromete el cuadrante superointerno de la mama derecha, "
            "de aproximadamente 30 mm.\n"
            "En la axila derecha se observan varias adenopatías de aspecto sospechoso.\n\n"
            "Impresión diagnóstica\n"
            "Neoplasia mamaria derecha que compromete piel y región areolar.\n"
            "Adenopatías axilares derechas de aspecto metastásico.\n"
            "BI-RADS US 5.\n\n" +
            _firma("Ignacio Tomas Bustos Carrasco") + _DESCARGO
        ),
        "espera": {"birads": 5, "categoria": None, "estado": "revision_extraccion",
                   "severidad": "critica"},
    },
    {
        "id": "seg_03_incoherencia_critica",
        "variante": "Sospecha con conducta insuficiente (incoherencia crítica)",
        "texto": (
            _encabezado("Pizarro Godoy, Camila Fernanda", "46A 6M", "16161616-6", "15-01-2026") +
            "Mamografía Bilateral\n"
            "Hallazgos:\n"
            "Microcalcificaciones pleomorfas agrupadas en cuadrante superior externo derecho.\n\n"
            "Impresión diagnóstica:\n"
            "Microcalcificaciones de aspecto altamente sospechoso.\n"
            "Birads 5.\n"
            "Se sugiere control anual.\n\n" +
            _firma("Alejandra Beatriz Riquelme Ponce") + _DESCARGO
        ),
        "espera": {"birads": 5, "categoria": "control_anual", "estado": "incoherente",
                   "severidad": "critica"},
    },

    # -----------------------------------------------------------------------
    # Grupo 5: privacidad
    # -----------------------------------------------------------------------
    {
        "id": "priv_01_nombre_en_linea_clinica",
        "variante": "Nombre del radiólogo e identificador pegados al texto clínico",
        "texto": (
            _encabezado("Alvarez Miranda, Rocio Alejandra", "59A 8M", "17171717-7", "16-01-2026") +
            "Mamografía Bilateral\n"
            "Hallazgos:\n"
            "Parénquima mamario heterogéneo. Sin lesiones sospechosas.\n\n"
            "Conclusión: examen sin hallazgos sospechosos. Birads 2. Se sugiere control "
            "anual. Informe validado por Dr. Sebastian Andres Molina Vera - RUT 18181818-8\n"
        ),
        "espera": {"birads": 2, "categoria": "control_anual", "estado": "coherente",
                   "sin_identificadores": True},
    },
]


# ---------------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------------

def _verificar_privacidad(texto_procesado: str) -> List[str]:
    """Devuelve la lista de identificadores que sobrevivieron a la limpieza."""
    import re
    problemas = []
    if re.search(r"\b\d{7,8}\s*-\s*[\dkK]\b", texto_procesado):
        problemas.append("RUT")
    if re.search(r"\bDr\s*\(?a?\)?\s*\.", texto_procesado, re.IGNORECASE):
        problemas.append("nombre de médico")
    if re.search(r"Nombre\s+Paciente", texto_procesado, re.IGNORECASE):
        problemas.append("campo de paciente")
    return problemas


def ejecutar(verbose: bool = True) -> Dict[str, Any]:
    """Corre la batería completa y devuelve el resumen."""
    import sys
    from pathlib import Path
    raiz = Path(__file__).resolve().parent.parent
    if str(raiz) not in sys.path:
        sys.path.insert(0, str(raiz))

    from src.predict import procesar_informe
    from src.recursos.limpieza_informe import limpiar_informe

    total = ok_birads = ok_categoria = ok_estado = ok_privacidad = 0
    fallos: List[str] = []

    if verbose:
        print("=" * 78)
        print("BATERÍA DE CASOS SINTÉTICOS — variantes de formato")
        print("=" * 78)
        print(f"{'ID':<30} {'BR':<5} {'Recomendación':<22} {'Estado'}")
        print("-" * 78)

    for caso in CASOS:
        total += 1
        r = procesar_informe(full_report=caso["texto"], informe_id=caso["id"])
        esp = caso["espera"]

        br = r["birads"]["valor"]
        cat = r["recomendacion"]["categoria_principal"]
        est = r["cotejo_acr"]["estado"]

        if br == esp["birads"]:
            ok_birads += 1
        else:
            fallos.append(f"{caso['id']}: BI-RADS esperado {esp['birads']}, obtenido {br}")

        if esp.get("categoria") is None or cat == esp["categoria"]:
            ok_categoria += 1
        else:
            fallos.append(f"{caso['id']}: categoría esperada {esp['categoria']}, obtenida {cat}")

        if esp.get("estado") is None or est == esp["estado"]:
            ok_estado += 1
        else:
            fallos.append(f"{caso['id']}: estado esperado {esp['estado']}, obtenido {est}")

        if esp.get("severidad"):
            sev = r["cotejo_acr"].get("severidad")
            if sev != esp["severidad"]:
                fallos.append(f"{caso['id']}: severidad esperada {esp['severidad']}, obtenida {sev}")

        procesado, _, _ = limpiar_informe(caso["texto"])
        problemas = _verificar_privacidad(procesado)
        if not problemas:
            ok_privacidad += 1
        else:
            fallos.append(f"{caso['id']}: identificadores no redactados: {', '.join(problemas)}")

        if verbose:
            print(f"{caso['id']:<30} {str(br):<5} {str(cat)[:21]:<22} {est}")

    if verbose:
        print("-" * 78)
        print(f"\n  BI-RADS correcto        : {ok_birads}/{total}")
        print(f"  Recomendación correcta  : {ok_categoria}/{total}")
        print(f"  Estado del cotejo        : {ok_estado}/{total}")
        print(f"  Sin identificadores      : {ok_privacidad}/{total}")
        if fallos:
            print(f"\n  FALLOS ({len(fallos)}):")
            for f in fallos:
                print(f"    - {f}")
        else:
            print("\n  Todos los casos pasan.")

    return {
        "total": total,
        "birads_ok": ok_birads,
        "categoria_ok": ok_categoria,
        "estado_ok": ok_estado,
        "privacidad_ok": ok_privacidad,
        "fallos": fallos,
    }


if __name__ == "__main__":
    import sys
    resultado = ejecutar()
    sys.exit(0 if not resultado["fallos"] else 1)
