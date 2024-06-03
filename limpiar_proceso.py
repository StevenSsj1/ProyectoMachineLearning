"""Limpiar los textos"""
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
#nltk.download('punkt')
import unicodedata
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
#nltk.download('stopwords')
# Lista de nombres comunes y títulos en español
titulos = ["sr", "sra", "dr", "dra", "lic", "ing", "prof", "secretario", "director", "ab", "senor", "senora"]
numeros_como_palabras = ["hora","minuto","uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa", "cien"]

# Lista de meses
meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# Lista de días de la semana
dias_semana = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# Función para eliminar nombres utilizando una lista de nombres comunes y títulos
def eliminar_nombres(texto):
    # Normalizar el texto para eliminar acentos y tildes
    texto = unicodedata.normalize('NFKD', texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])

    # Convertir el texto a minúsculas
    texto = texto.lower()

    for hora in numeros_como_palabras:
        texto = re.sub(r'\b{}\b \w+'.format(hora), '', texto)

    # Eliminar títulos y los nombres que les siguen
    for mes in meses:
        texto = re.sub(r'\b{}\b \w+'.format(mes), '', texto)

    for dia in dias_semana:
        texto = re.sub(r'\b{}\b \w+'.format(dia), '', texto)

    for titulo in titulos:
        texto = re.sub(r'\b{}\b \w+'.format(titulo), '', texto)

    return texto

def normalizar_texto(texto):
    # Normalizar caracteres eliminando acentos y tildes
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

