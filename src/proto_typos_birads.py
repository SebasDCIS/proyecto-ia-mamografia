"""
PROTOTIPO — capa de tolerancia a typos para el token BI-RADS.
NO modifica buscador_birads.py. Es un experimento aislado.

Principio de diseño (simétrico al módulo 2):
  - Tolerancia de edición SOLO sobre el token "birads".
  - NUNCA sobre el dígito: "BI-RADS 2" y "BI-RADS 5" están a distancia 1.
  - Confusión de caracteres SOLO en la posición del dígito (O->0, l/I->1, S->5),
    porque ahí un carácter no numérico es inequívocamente un error de OCR/tipeo.
"""
import re
import unicodedata

CANONICO = "birads"
UMBRAL_TOKEN = 1          # distancia máxima de edición sobre el token
CONFUSION_DIGITO = {"o": "0", "O": "0", "l": "1", "I": "1", "i": "1", "S": "5", "s": "5"}


def damerau_levenshtein(a, b):
    """Distancia de edición con transposición de adyacentes."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# Candidatos: secuencias que EMPIEZAN con b y tienen largo plausible.
# El guion y el espacio interno se toleran (bi-rads, bi rads, birads).
PATRON_CANDIDATO = re.compile(r"\bb[a-z]{1,3}[\s\-]?r[a-z]{1,4}s?\b", re.IGNORECASE)


def buscar_token_birads_con_typos(texto):
    """Devuelve [(inicio, fin, token_original, distancia)] de tokens ~ 'birads'."""
    hits = []
    for m in PATRON_CANDIDATO.finditer(texto):
        crudo = m.group(0)
        norm = _sin_tildes(re.sub(r"[\s\-]", "", crudo)).lower()
        d = damerau_levenshtein(norm, CANONICO)
        if d <= UMBRAL_TOKEN:
            hits.append((m.start(), m.end(), crudo, d))
    return hits


def leer_digito_despues(texto, fin_token, ventana=14):
    """Lee el valor BI-RADS tras el token. Tolera O->0, l->1, S->5.

    Devuelve (valor, lexema, tipo) o None.
    tipo: 'digito' | 'confusion_caracter'
    """
    cola = texto[fin_token: fin_token + ventana]
    # Salta separadores tipicos: ®, :, -, espacios, "us", "(segun la acr)"
    m = re.match(r"[\s®:\-–]*(?:us|mx|rm)?[\s®:\-–]*", cola, re.IGNORECASE)
    resto = cola[m.end():] if m else cola
    if not resto:
        return None

    ch = resto[0]
    if ch.isdigit():
        v = int(ch)
        return (v, ch, "digito") if 0 <= v <= 6 else None

    if ch in CONFUSION_DIGITO:
        # Solo si NO es el inicio de una palabra (evita "birads sin hallazgos")
        siguiente = resto[1] if len(resto) > 1 else " "
        if siguiente.isalpha():
            return None
        v = int(CONFUSION_DIGITO[ch])
        return (v, ch, "confusion_caracter") if 0 <= v <= 6 else None

    return None


def buscar_birads_tolerante(texto):
    """Busqueda de rescate: solo para cuando el buscador oficial no encontro nada."""
    out = []
    for ini, fin, crudo, d in buscar_token_birads_con_typos(texto):
        lec = leer_digito_despues(texto, fin)
        if lec is None:
            continue
        valor, lexema, tipo = lec
        out.append({
            "valor": valor,
            "token": crudo,
            "distancia_token": d,
            "lexema_valor": lexema,
            "tipo_lectura": tipo,
            "posicion": ini,
            "posicion_relativa": ini / max(len(texto), 1),
            "contexto": texto[max(0, ini - 60): fin + 30].strip(),
        })
    return out
