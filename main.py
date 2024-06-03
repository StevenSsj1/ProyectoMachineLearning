import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
from PyPDF2 import PdfReader
import io
import re
from prediccion import clasificar_documento
import json
import pandas as pd
import redis
import dash_bootstrap_components as dbc
from sklearn.feature_extraction.text import CountVectorizer
from Filtrp_pdf import *

r = redis.StrictRedis(host='localhost', port=6379, db=0)

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

promedio_tfidf_culpables_recuperado = json.loads(r.get('promedio_tfidf_culpables'))
promedio_tfidf_inocentes_recuperado = json.loads(r.get('promedio_tfidf_inocentes'))

promedio_tfidf_culpables_df = pd.DataFrame.from_dict(promedio_tfidf_culpables_recuperado, orient='index')
promedio_tfidf_inocentes_df = pd.DataFrame.from_dict(promedio_tfidf_inocentes_recuperado, orient='index')

v1 = promedio_tfidf_culpables_df[0].values
v2 = promedio_tfidf_inocentes_df[0].values

vectorizer = CountVectorizer()

cdi = json.loads(r.get('tokens_inocentes'))
cdc = json.loads(r.get('tokens_culpables'))

X = vectorizer.fit_transform(cdi + cdc)

def obtener_nombre_archivo(contents, filename):
    return html.Div([
        html.H5(filename)
    ])

# Función para analizar el contenido del archivo PDF
def parse_contents(contents, filename):
    # Obtener el contenido del archivo y convertirlo a base64
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Leer el archivo PDF y extraer su texto
    pdf = PdfReader(io.BytesIO(decoded))
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    
    # Eliminar pies de página
    texto_sin_pie = eliminar_pie_pagina(text)
    
    # Encontrar títulos y texto entre ellos
    texto_sin_titulos = encontrar_titulos_y_texto(texto_sin_pie)
    
    # Si no se encontraron títulos ni pies de página, devolver todo el texto sin procesar
    if not texto_sin_titulos and not re.search(r'\bPágina \d+ de \d+\b', text):
        texto_completo = text
    else:
        # Filtrar la información del veredicto para cada bloque de texto
        texto_filtrado = [filter_verdict_information(bloque[1]) for bloque in texto_sin_titulos]
    
        # Unir todos los textos filtrados en un solo string
        texto_completo = '\n'.join(texto_filtrado)

    return html.Div([
        html.H5(filename),
        html.Pre(clasificar_documento(texto_completo, vectorizer, v1, v2))
    ])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
# Definir el diseño de la aplicación
app.layout = html.Div(style={'textAlign': 'center', 'fontFamily': 'Arial'}, children=[
    html.Img(src='path_to_logo.png', alt='Universidad Salesiana', style={'height': '100px'}),
    html.H1(children='Proyecto ML', style={'color': '#007bff'}),
    
    html.Div(children='''
        Esta es una aplicación web de ejemplo utilizando Dash.
    ''', style={'fontSize': '18px'}),
    html.Div([dcc.Location(id="url"), sidebar, content]),
]),


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.Div([
            html.P("Oh cool, this is page 2!"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Arrastra y suelta o ',
                    html.A('selecciona un archivo PDF', style={'color': '#007bff', 'textDecoration': 'underline'})
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '20px'
                },
                # Permitir subir solo archivos PDF
                accept='.pdf'
            ),
            html.Div(id='output-data-upload', style={'marginTop': '20px'}),
            html.Div(id='upload-progress-container', children=[
                html.Div(id='upload-progress'),
                html.Div(id='upload-output')
            ]),
        ])
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )
    
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              prevent_initial_call=True)
def update_output(contents, filename):
    if contents is not None:
        children = parse_contents(contents, 'PDF cargado')
        nombre_archivo = obtener_nombre_archivo(contents, filename)
        return nombre_archivo, children

if __name__ == '__main__':
    app.run_server(debug=True)
