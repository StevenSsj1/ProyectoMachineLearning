import os
import re
import pandas as pd
from PyPDF2 import PdfReader
import nltk
nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.metrics.pairwise import cosine_similarity
import redis
import json
nltk.download('punkt')
import unicodedata
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

# Función para extraer texto de un PDF
def extraer_texto(pdf_path):
    texto = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            texto += page.extract_text()
    return texto

# Función para encontrar títulos y el texto entre ellos
def encontrar_titulos_y_texto(texto):
    # Patrón para detectar los títulos
    patron_titulo = r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2} [A-ZÁÉÍÓÚÑ\s]+ \([A-ZÁÉÍÓÚÑ\s]+\))'
    bloques = re.split(patron_titulo, texto)
    # Eliminar el primer elemento vacío
    bloques = bloques[1:]
    # Agrupar títulos y texto entre ellos
    bloques = [bloques[i:i+2] for i in range(0, len(bloques), 2)]
    return bloques

# Función para eliminar fechas, números y caracteres no alfanuméricos
def eliminar_fechas_numeros_no_alfanumericos(texto):
    patron_fechas = r'\d{2}/\d{2}/\d{4}'  # Coincide con fechas en formato dd/mm/yyyy
    texto_sin_fechas = re.sub(patron_fechas, '', texto)

    patron_numeros = r'\d+'  # Coincide con números
    texto_sin_numeros = re.sub(patron_numeros, '', texto_sin_fechas)

    texto_filtrado = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', texto_sin_numeros)
    return texto_filtrado

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

# Directorio con archivos PDF
pdf_directory = "/home/administrador/Documentos/ProyectoMLP1/Documentos"

# Listar todos los archivos PDF en el directorio
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

data = []

# Procesar cada archivo PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    # Extraer texto del PDF
    texto_pdf = extraer_texto(pdf_path)

    # Encontrar títulos y el texto entre ellos
    titulos_y_texto = encontrar_titulos_y_texto(texto_pdf)
    for titulo, texto in titulos_y_texto:
        texto_sin_pie = eliminar_pie_pagina(texto)
        texto_filtrado = filter_verdict_information(texto_sin_pie)
        texto_final = eliminar_fechas_numeros_no_alfanumericos(texto_filtrado)
        data.append((pdf_file, titulo, texto_final.strip()))

# Crear el DataFrame
df = pd.DataFrame(data, columns=['NombreArchivo', 'Título', 'Texto'])

# Guardar el DataFrame en un archivo CSV
df.to_csv('documento_procesado.csv', index=False)

# Filtrar filas donde el título contenga ciertas palabras clave
condicion = df['Título'].str.contains('RESOLUCION') | df['Título'].str.contains('AUTO RESOLUTIVO')
filas_seleccionadas = df[condicion]

# Función para asignar sentencia basada en el número del archivo
def asignar_sentencia(nombre_archivo):
    numero = int(nombre_archivo.split('.')[0])
    return 0 if 0 <= numero <= 9 else 1

# Unir textos de filas con títulos repetidos y asignar sentencia
filas_unidas = df.groupby('NombreArchivo')['Texto'].apply(' '.join).reset_index()
filas_unidas['sentencia'] = filas_unidas['NombreArchivo'].apply(asignar_sentencia)

# Convertir la columna 'NombreArchivo' a un tipo que pandas puede ordenar correctamente
filas_unidas['NumeroArchivo'] = filas_unidas['NombreArchivo'].str.extract(r'(\d+)', expand=False).astype(int)

# Ordenar el DataFrame por 'NumeroArchivo'
filas_unidas = filas_unidas.sort_values(by='NumeroArchivo')

# Asignar valores de sentencia: 1 para los primeros 10, 0 para los siguientes 10
# Nota: Asegúrate de que hay al menos 20 archivos, de lo contrario, este paso dará error.
if len(filas_unidas) >= 20:
    filas_unidas['sentencia'] = [1] * 10 + [0] * 10
else:
    raise ValueError("No hay suficientes archivos para asignar sentencias. Se requieren al menos 20 archivos.")

# Eliminar la columna auxiliar 'NumeroArchivo'
df_sorted = filas_unidas.drop(columns=['NumeroArchivo'])
filas_unidas_var = df_sorted['Texto']
print(filas_unidas)
r = redis.StrictRedis(host='localhost', port=6379, db=0)
# Imprimir el DataFrame ordenado y con la columna 'sentencia'

# Serializar el DataFrame final y guardarlo en Redis
df_final_serializado = pickle.dumps(df_sorted)
r.set('documento_procesado_final', df_final_serializado)

# Recuperar el DataFrame final de Redis (opcional)
df_recuperado_serializado = r.get('documento_procesado_final')
df_recuperado = pickle.loads(df_recuperado_serializado)


# Lista de nombres comunes y títulos en español
titulos = ["sr", "sra", "dr", "dra", "lic", "ing", "prof", "secretario", "director", "ab", "senor", "senora"]
numeros_como_palabras = ["hora","minuto","uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa", "cien"]

# Lista de meses
meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# Lista de días de la semana
dias_semana = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# Función para eliminar nombres utilizando una lista de nombres comunes y títulos
def eliminar_nombres(texto):
    #Normalizar el texto para eliminar acentos y tildes
    texto = unicodedata.normalize('NFKD', texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])

    #Convertir el texto a minúsculas
    texto = texto.lower()

    for hora in numeros_como_palabras:
        texto = re.sub(r'\b{}\b \w+'.format(hora), '', texto)

    #Eliminar títulos y los nombres que les siguen
    for mes in meses:
        texto = re.sub(r'\b{}\b \w+'.format(mes), '', texto)

    for dia in dias_semana:
        texto = re.sub(r'\b{}\b \w+'.format(dia), '', texto)

    for titulo in titulos:
        texto = re.sub(r'\b{}\b \w+'.format(titulo), '', texto)

    return texto

def normalizar_texto(texto):
    #Normalizar caracteres eliminando acentos y tildes
    texto = unicodedata.normalize('NFKD', texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])
    return texto

def clean_text(*args):
    regex_date = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'  # Expresión regular para eliminar fechas en formato dd/mm/yyyy
    regex_number = r'\b\d+\b'  # Expresión regular para eliminar números
    regex_non_alnum = r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]'  # Expresión regular para eliminar caracteres no alfanuméricos
    stop_words = set(stopwords.words('spanish'))  # Obtener palabras vacías en español
    stemmer = PorterStemmer()  # Inicializar el objeto para realizar la derivación (stemming)
    cleaned_texts = []  # Lista para almacenar los textos limpios

    for text in args:
        cleaned_text = normalizar_texto(text)  # Normalizar texto para eliminar acentos y tildes
        cleaned_text = eliminar_nombres(cleaned_text)  # Eliminar nombres comunes y títulos
        cleaned_text = re.sub(regex_date, '', cleaned_text)  # Eliminar fechas
        cleaned_text = re.sub(regex_number, '', cleaned_text)  # Eliminar números
        cleaned_text = re.sub(regex_non_alnum, ' ', cleaned_text)  # Eliminar caracteres no alfanuméricos
        cleaned_text = cleaned_text.lower()  # Convertir texto a minúsculas
        tokens = word_tokenize(cleaned_text)  # Tokenizar texto
        tokens = [word for word in tokens if word not in stop_words]  # Eliminar palabras vacías
        tokens = [stemmer.stem(word) for word in tokens]  # Aplicar derivación a las palabras
        tokens = [word for word in tokens if len(word) > 1]  # Eliminar palabras de una sola letra
        cleaned_texts.append(" ".join(tokens))  # Unir las palabras limpias en un solo texto

    return cleaned_texts

d =clean_text(*filas_unidas_var)


# Supongamos que tienes dos listas de textos preprocesados: documentos_culpables y documentos_inocentes
documentos_culpables = filas_unidas.loc[0:9, 'Texto'].tolist()  # Llena esto con tus textos preprocesados de documentos culpables
documentos_inocentes = filas_unidas.loc[10:, 'Texto'].tolist()  # Llena esto con tus textos preprocesados de documentos inocentes

cdc = clean_text(*documentos_culpables)
cdi = clean_text(*documentos_inocentes)

# Guardar la lista en Redis
r.set('tokens_culpables', json.dumps(cdc))
r.set('tokens_inocentes', json.dumps(cdi))

# Recuperar la lista de Redis
inocentes = json.loads(r.get('tokens_inocentes'))

# Unir ambos conjuntos para ajustar el vectorizador a todos los documentos
todos_los_documentos =  cdc  + cdi


# # Crear un TF-IDF vectorizer y ajustar a todos los documentos
vectorizer = TfidfVectorizer()  # Puedes ajustar el número de características según sea necesario
vectorizer.fit(todos_los_documentos)
# Transformar cada conjunto de documentos

# # Transformar cada conjunto de documentos
tfidf_culpables = vectorizer.transform(documentos_culpables)
tfidf_inocentes = vectorizer.transform(documentos_inocentes)

#Obtener los nombres de las características
palabras = vectorizer.get_feature_names_out()

# Convertir las matrices TF-IDF a DataFrames para mejor manejo
df_tfidf_culpables = pd.DataFrame(tfidf_culpables.toarray(), columns=palabras)
df_tfidf_inocentes = pd.DataFrame(tfidf_inocentes.toarray(), columns=palabras)

# Promediar los TF-IDF para cada palabra en ambos grupos
promedio_tfidf_culpables = df_tfidf_culpables.mean(axis=0)
promedio_tfidf_inocentes = df_tfidf_inocentes.mean(axis=0)

# Función para calcular la similitud coseno entre un nuevo documento y los promedios TF-IDF de culpables e inocentes
def calcular_similitud(nuevo_documento, vectorizer, promedio_tfidf_culpables, promedio_tfidf_inocentes):
    # Preprocesar el nuevo documento
    nuevo_documento_procesado = clean_text(nuevo_documento)[0]  # Ajustar para procesar un solo documento

    # Transformar el documento usando el vectorizador TF-IDF
    nuevo_documento_tfidf = vectorizer.transform([nuevo_documento_procesado])

    # Calcular la similitud coseno con los promedios de culpables e inocentes
    similitud_culpables = cosine_similarity(nuevo_documento_tfidf, [promedio_tfidf_culpables])
    similitud_inocentes = cosine_similarity(nuevo_documento_tfidf, [promedio_tfidf_inocentes])

    return similitud_culpables[0][0], similitud_inocentes[0][0]

# Crear una matriz de distancias
matriz_distancias = np.zeros((20, 2))

# Iterar sobre los documentos en el vector d del 0 al 19
for i in range(20):
    nuevo_documento = d[i]
    similitud_culpables, similitud_inocentes = calcular_similitud(nuevo_documento, vectorizer, promedio_tfidf_culpables, promedio_tfidf_inocentes)
    matriz_distancias[i, 0] = similitud_culpables
    matriz_distancias[i, 1] = similitud_inocentes

# Convertir la matriz de distancias a un DataFrame para mejor visualización
df_matriz_distancias = pd.DataFrame(matriz_distancias, columns=['Similitud Culpables', 'Similitud Inocentes'])

# # Imprimir la matriz de distancias
print(df_matriz_distancias)

# Determinar la clasificación basada en la similitud más alta
resultados = []
for i in range(20):
    if matriz_distancias[i, 0] > matriz_distancias[i, 1]:
        resultado = "Culpable"
    else:
        resultado = "Inocente"
    resultados.append((i, resultado))

# Imprimir los resultados
for idx, resultado in resultados:
    print(f"El documento {idx} es: {resultado}")


# Conectarse a Redis

# Convertir Series a diccionarios
promedio_tfidf_culpables_dict = promedio_tfidf_culpables.to_dict()
promedio_tfidf_inocentes_dict = promedio_tfidf_inocentes.to_dict()

# Guardar los diccionarios en Redis
r.set('promedio_tfidf_culpables', json.dumps(promedio_tfidf_culpables_dict))
r.set('promedio_tfidf_inocentes', json.dumps(promedio_tfidf_inocentes_dict))

# Para recuperar los datos desde Redis
promedio_tfidf_culpables_recuperado = json.loads(r.get('promedio_tfidf_culpables'))
promedio_tfidf_inocentes_recuperado = json.loads(r.get('promedio_tfidf_inocentes'))

# Convertir los datos recuperados a DataFrame si es necesario
promedio_tfidf_culpables_df = pd.DataFrame.from_dict(promedio_tfidf_culpables_recuperado, orient='index')
promedio_tfidf_inocentes_df = pd.DataFrame.from_dict(promedio_tfidf_inocentes_recuperado, orient='index')

print(promedio_tfidf_culpables_df)