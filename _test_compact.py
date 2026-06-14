"""Unit tests for _compact_heading + glossary wiring, using the real failure
cases found in the baseline scan."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from translator import _compact_heading, _has_dup_prefixes

cases = [
    # (input, allow_word_echo, expected)
    # 1. Paraphrase restart after sentence boundary (echo across boundary)
    ("¿Quién debería leer este libro?, quién deberia leerlo.", True,
     "¿Quién debería leer este libro?"),
    ("x Tabla de Contenidos.x Cuadro del contenido", True,
     "x Tabla de Contenidos."),
    ("Soluciones para Replicación Lag.: Replication Solutions for", True,
     "Soluciones para Replicación Lag."),
    # 2. Legitimate internal boundary, no echo → kept
    ("Publicado por O'Reilly Media, Inc. 1005 Gravenstein Highway Norte", True,
     "Publicado por O'Reilly Media, Inc. 1005 Gravenstein Highway Norte"),
    # 3. Content-word echo inside heading
    ("Almacenamiento Orientado Columna de la columna", True,
     "Almacenamiento Orientado Columna"),
    ("Operación Multi Región de Múltiples Regiones", True,
     "Operación Multi Región"),
    ("Arquitectura del sistema nativo de la nube nativa", True,
     "Arquitectura del sistema nativo de la nube"),
    # 4. Echo disabled (source had legit repeats) → untouched
    ("Memoria compartida, disco compartido y arquitecturas de nada Compartida",
     False,
     "Memoria compartida, disco compartido y arquitecturas de nada Compartida"),
    # 5. Long output (>10 words) → word-echo skipped
    ("La nube versus el alojamiento web cloud frente a uno mismo hosting", True,
     "La nube versus el alojamiento web cloud frente a uno mismo hosting"),
    # 6. Clean headings stay clean
    ("Latencia y tiempo de respuesta", True, "Latencia y tiempo de respuesta"),
    ("Problemas de los sistemas distribuidos", True,
     "Problemas de los sistemas distribuidos"),
    # 7. Regresión fase1c: repetición legítima del sustantivo en "X versus Y"
    #    (repite a mitad del título, no al final) → NO cortar
    ("Sistemas operativos versus sistemas analíticos 3", True,
     "Sistemas operativos versus sistemas analíticos 3"),
    # 8. Regresión fase1c: referencia bibliográfica con autor duplicado por el
    #    modelo — el boundary-cut dejaría solo "Edgar F." (<40%) → NO cortar
    ("Edgar F. [ 5 ] Edgar Edgard F. Codd, y C. T. Salley. Proporción de OLAP "
     "a los analistas del usuario", True,
     "Edgar F. [ 5 ] Edgar Edgard F. Codd, y C. T. Salley. Proporción de OLAP "
     "a los analistas del usuario"),
]

ok = 0
for text, allow, expected in cases:
    got = _compact_heading(text, allow)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} {text[:55]!r}")
    if got != expected:
        print(f"     esperado: {expected!r}")
        print(f"     obtenido: {got!r}")

dup_cases = [
    ("Shared-Memory, Shared-Disk, and Shared-Nothing Architectures", True),
    ("Data storage and data systems", True),
    ("Trade Offs in Data Systems Architecture", False),
    ("Understanding Load", False),
]
for text, expected in dup_cases:
    got = _has_dup_prefixes(text)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} _has_dup_prefixes({text[:45]!r}) = {got}")

from translator import _split_footer

footer_cases = [
    ("Operational Versus Analytical Systems | 3",
     ("", "Operational Versus Analytical Systems", " | 3")),
    ("x | Table of Contents", ("x | ", "Table of Contents", "")),
    ("xviii | Preface", ("xviii | ", "Preface", "")),
    ("a | b normal pipe sin numero", None),
    ("Texto normal sin pipe", None),
]
for text, expected in footer_cases:
    got = _split_footer(text)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} _split_footer({text[:40]!r}) = {got}")

from translator import _split_sentences, _recover_unk_heads, _mask_cross_refs

sent_cases = [
    # Iniciales NO parten la oración (causa raíz del garbage bibliográfico)
    ("Edgar F. Codd, S. B. Codd, and C. T. Salley wrote it.",
     ["Edgar F. Codd, S. B. Codd, and C. T. Salley wrote it."]),
    # Abreviaturas protegidas
    ("Use a database, e.g. PostgreSQL, for storage.",
     ["Use a database, e.g. PostgreSQL, for storage."]),
    # Oraciones normales sí se parten
    ("First sentence. Second sentence.",
     ["First sentence.", "Second sentence."]),
]
for text, expected in sent_cases:
    got = _split_sentences(text)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} _split_sentences({text[:45]!r}) = {got}")

unk_cases = [
    ("El ⁇ndice está al final", "El Índice está al final"),
    ("⁇ndices Multicolumna", "Índices Multicolumna"),
    ("El ⁇ltimo capítulo", "El Último capítulo"),
    ("⁇xyzfragmento raro", "⁇xyzfragmento raro"),  # sin match → intacto
]
for text, expected in unk_cases:
    got = _recover_unk_heads(text)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} _recover_unk_heads({text[:35]!r}) = {got!r}")

masked, restores = _mask_cross_refs(
    'Compare both (see "Operational Versus Analytical Systems" on page 3).')
xref_ok = (masked == 'Compare both (901).'
           and restores == [('(901)',
                '(ver "Sistemas operacionales versus analíticos" en la página 3)')])
if xref_ok:
    ok += 1
print(f"{'OK ' if xref_ok else 'FAIL'} _mask_cross_refs paren = {masked!r} {restores}")

masked2, restores2 = _mask_cross_refs('As we saw, see "Unknown Title Here" on page 9 for more.')
xref2_ok = (masked2 == 'As we saw, (901) for more.'
            and restores2 == [('(901)', 'ver "Unknown Title Here" en la página 9')])
if xref2_ok:
    ok += 1
print(f"{'OK ' if xref2_ok else 'FAIL'} _mask_cross_refs bare = {masked2!r}")

from translator import _is_author_sentence, _truncate_output

author_cases = [
    ("Edgar F. Codd, S. B. Codd, and C. T. Salley.", True),
    ("Rui Pedro Machado and Helder Russa.", True),
    ("Betsy Beyer, Jennifer Petoff, Chris Jones, and Niall Richard Murphy.", True),
    ("Providing OLAP to User-Analysts: An IT Mandate.", False),  # tiene "to"
    ("The spectrum of decisions on outsourcing software.", False),
]
for text, expected in author_cases:
    got = _is_author_sentence(text)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} _is_author_sentence({text[:48]!r}) = {got}")

trunc_cases = [
    # Iniciales en el OUTPUT no deben truncar
    ("Edgar F. Codd, S. B. Codd y C. T. Salley escribieron esto",
     "Edgar F. Codd, S. B. Codd y C. T. Salley escribieron esto"),
    # El truncado legítimo de restart sigue funcionando
    ("Edición. 2a edición de la obra", "Edición."),
]
for text, expected in trunc_cases:
    got = _truncate_output(text)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} _truncate_output({text[:45]!r}) = {got!r}")

from translator import _collapse_repeated_ngrams

ngram_cases = [
    ("se detuvo en seco se detuvo en seco frente a la puerta",
     "se detuvo en seco frente a la puerta"),
    ("la luz era tenue la luz era tenue", "la luz era tenue"),
    # Triple repetición → una sola
    ("muy lejos muy lejos muy lejos de aquí", "muy lejos de aquí"),
    # Sin repetición → intacto
    ("una frase normal sin ninguna repeticion interna",
     "una frase normal sin ninguna repeticion interna"),
    # Repetición NO adyacente → intacto (legítima)
    ("el pintor miró al cielo y el pintor sonrió",
     "el pintor miró al cielo y el pintor sonrió"),
]
for text, expected in ngram_cases:
    got = _collapse_repeated_ngrams(text)
    status = "OK " if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"{status} _collapse_repeated_ngrams({text[:42]!r}) = {got!r}")

total = (len(cases) + len(dup_cases) + len(footer_cases) + len(sent_cases)
         + len(unk_cases) + 2 + len(author_cases) + len(trunc_cases)
         + len(ngram_cases))
print(f"\n{ok}/{total} pasaron")
sys.exit(0 if ok == total else 1)

# El resultado esperado es que todos los casos pasen xxd