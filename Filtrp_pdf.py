import re
# Función para encontrar títulos y el texto entre ellos
def encontrar_titulos_y_texto(texto):
    # Patrón para detectar los títulos
    patron_titulo = r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2} [A-Z\s]+ \([A-Z\s]+\))'
    bloques = re.split(patron_titulo, texto)
    # Eliminar el primer elemento vacío
    bloques = bloques[1:]
    # Agrupar títulos y texto entre ellos
    bloques = [bloques[i:i+2] for i in range(0, len(bloques), 2)]
    return bloques

# Función para eliminar los pies de página
def eliminar_pie_pagina(texto):
    # Patrón para eliminar el pie de página
    patron_pie_pagina = r'Página \d+ de \d+'
    texto_sin_pie = re.sub(patron_pie_pagina, '', texto)
    return texto_sin_pie

# Función para filtrar la información del veredicto
def filter_verdict_information(texto):
    # Patrones para excluir información sobre el veredicto
    patrones_veredicto = [
        r'culpable',  # coincide con "culpable"
        r'inocente',  # coincide con "inocente"
        r'verdicto',  # coincide con "veredicto"
        r'sentencia',  # coincide con "sentencia"
        r'condena',  # coincide con "condena"
        r'absuelto',  # coincide con "absuelto"
        r'libertad',  # coincide con "libertad"
        r'culpabilidad',  # coincide con "culpabilidad"
        r'inocencia'  # coincide con "inocencia"
        # Agregar más patrones según sea necesario
    ]

    # Patrón para eliminar correos electrónicos
    patron_correo = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Combinar los patrones en una sola expresión regular
    patron_combinado = re.compile('|'.join(patrones_veredicto + [patron_correo]), re.IGNORECASE)

    # Eliminar todas las coincidencias
    texto_filtrado = patron_combinado.sub('', texto)

    return texto_filtrado