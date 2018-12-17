import dash_core_components as dcc
import dash_html_components as html

from App import app

# Configure navbar menu
nav_menu = html.Div([
    html.Div([
        ## NavBar header
        html.Div([
            html.A('Style Transfer Learning', className='navbar-brand')
        ], className='navbar-header'),      
        ## NavBar sub-headers
        html.Ul([
            html.Li([html.A('Home', href='/Home')]),
            html.Li([html.A('DIY', href='/DIY')]),
            html.Li([html.A('WebCam', href='/WebCam')]),
            html.Li([html.A('About', href='/About')], className='active')
        ], className='nav navbar-nav'),
        ## NavBar rignt-aligned
        html.Ul([
            html.Li([
                html.A([html.I(className="fa fa-github")], href='https://github.com/TZstatsADS/Spring2018-Project2-Group1')
            ])
        ], className="nav navbar-nav navbar-right")
    ], className="container-fluid")
], className="navbar navbar-inverse")

## nav bar
nav_bar = html.Nav([
    html.Div([
        html.Div([
            html.Button([
                html.Span(className="icon-bar"),
                html.Span(className="icon-bar"),
                html.Span(className="icon-bar")
            ], className="navbar-toggle", type="button", **{'data-toggle': 'collapse'}, **{'data-target': '#myNavbar'}),
            html.A('Neural Style Transfer', href="#myPage", className="navbar-brand")
        ], className="navbar-header"),
        html.Div([
            html.Ul([
                html.Li([html.A('Home', href='/Home')]),
                html.Li([html.A('DIY', href='/DIY')]),
                html.Li([html.A('WebCam', href='/WebCam')]),
                html.Li([html.A('About', href='/About')])
            ], className="nav navbar-nav navbar-right")
        ], className="collapse navbar-collapse", id="myNavbar")
    ], className="container")
], className="navbar navbar-default navbar-fixed-top")

## team
team = html.Div([
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('hl.jpg'), className="img-circle person"),
            html.P(html.Strong('Hongyu Li')),
            html.P('hl3099@columbia.edu')
        ], className="thumbnail")
    ], className='col-sm-3'),
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('ph.jpg'), className="img-circle person"),
            html.P(html.Strong('Peifeng Hong')),
            html.P('ph2534@columbia.edu')
        ], className="thumbnail")
    ], className='col-sm-3'),
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('jz.jpg'), className="img-circle person"),
            html.P(html.Strong('Jia Zheng')),
            html.P('jz2891@columbia.edu')
        ], className="thumbnail")
    ], className='col-sm-3'),
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('dw.jpg'), className="img-circle person"),
            html.P(html.Strong('Di Wu')),
            html.P('dw2794@columbia.edu')
        ], className="thumbnail")
    ], className='col-sm-3')        
], className="row text-center")

layout = html.Div([
    nav_bar,
    team
])