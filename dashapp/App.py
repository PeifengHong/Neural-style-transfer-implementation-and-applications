import dash

external_scripts = ["https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js",
                    "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"]

external_stylesheets = ["https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"]

app = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
server = app.server

app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True