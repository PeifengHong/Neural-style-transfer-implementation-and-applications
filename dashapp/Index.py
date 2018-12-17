import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

#Eager Mode
import tensorflow as tf
tf.enable_eager_execution()   
import tensorflow.contrib.eager as tfe

from App import app
import Home, DIY, WebCam, About

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if   pathname == '/DIY':  return DIY.layout
    elif pathname == '/WebCam':  return WebCam.layout
    elif pathname == '/About':  return About.layout
    else: return Home.layout

if __name__ == '__main__':
    app.run_server(debug=True)