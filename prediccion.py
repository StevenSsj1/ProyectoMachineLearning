import pandas as pd
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from limpiar_proceso import clean_text
import redis
import json


r = redis.StrictRedis(host='localhost', port=6379, db=0)

# Función para extraer texto de un PDF
def extraer_texto(pdf_path):
    texto = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            texto += page.extract_text()
    return texto


# Crear un vectorizador de conteo para convertir tokens en una matriz
vectorizer = CountVectorizer()

cdi = json.loads(r.get('tokens_inocentes'))
cdc = json.loads(r.get('tokens_culpables'))

X = vectorizer.fit_transform(cdi + cdc)

# Función para calcular la similitud coseno entre un nuevo documento y los promedios TF-IDF de culpables e inocentes
def calcular_similitud(nuevo_documento, vectorizer, promedio_tfidf_culpables, promedio_tfidf_inocentes):
    # Preprocesar el nuevo documento
    nuevo_documento_procesado = clean_text(nuevo_documento)[0]  # Ajustar para procesar un solo documento

    # Transformar el documento usando el vectorizador TF-IDF
    nuevo_documento_tfidf = vectorizer.transform([nuevo_documento_procesado])

    # Calcular la similitud coseno con los promedios de culpables e inocentes
    similitud_culpables = cosine_similarity(nuevo_documento_tfidf, [promedio_tfidf_culpables])
    similitud_inocentes = cosine_similarity(nuevo_documento_tfidf, [promedio_tfidf_inocentes])
    es_culpable = similitud_culpables.mean() > similitud_inocentes.mean() 
    return es_culpable

# Función para clasificar un nuevo documento
def clasificar_documento(nuevo_documento, vectorizer, promedio_tfidf_culpables, promedio_tfidf_inocentes):
    similitud_culpables = calcular_similitud(nuevo_documento, vectorizer, promedio_tfidf_culpables, promedio_tfidf_inocentes)
    return "El documento es culpable." if similitud_culpables  else "El documento es inocente."

# Ejemplo de uso: ingresa la ruta de tu nuevo documento PDF aquí

ruta_nuevo_documento = "/home/administrador/Documentos/ProyectoMLP1/expel_07710202000292_22278534_02062024.pdf"
texto_nuevo_documento = extraer_texto(ruta_nuevo_documento)

# resultado = clasificar_documento(texto_nuevo_documento, vectorizer, v1, v2)
# print(f"El documento es: {resultado}")
