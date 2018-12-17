import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from App import app
import base64
import io
import cv2

app.scripts.config.serve_locally = True

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
                html.Li([html.A('About', href='/About')]),
                html.Li([html.A([
                    html.I(className="fa fa-github")
                ], href='https://github.com/PeifengHong/Neural-style-transfer-implementation-and-applications/')])
            ], className="nav navbar-nav navbar-right")
        ], className="collapse navbar-collapse", id="myNavbar")
    ], className="container")
], className="navbar navbar-default navbar-fixed-top")

## upload image
upload_img = html.Div([
    html.Br(),
    html.H1('Style Transfer An Image'),
    html.P('Use Deep Learning to Automatically Style Tranfer An Image'),
    html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '50%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '0 auto'
                    },
                    multiple=True
                ),
                html.Br(),
                html.Div(id='output-image-upload'),
            ]),  
    ], style={
        'textAlign': 'center',
    })
], className="jumbotron text-center")

def parse_contents(contents):
    content_type, content_string = contents.split(',')      
    decoded = base64.b64decode(content_string)
    img = np.array(Image.open(io.BytesIO(decoded)))
    imgsize = img.shape[0:2]
    pred_img = style_transfer_test_data(img,512,512)
    pred_img = cv2.resize(pred_img,imgsize[::-1])
    pred_img = Image.fromarray(pred_img,'RGB')
    buff = io.BytesIO()
    pred_img.save(buff,format='png')
    encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")
    HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '
    show_img = html.Div([
        html.Img(src=contents,width="49%"),
        html.Img(id=f'img-{id}',
                 src=HTML_IMG_SRC_PARAMETERS + encoded_img, 
                 width="49%")
    ])
    return show_img

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')])

def update_output(contents):
    if contents is not None:
        children = [parse_contents(contents[0])]
        return children

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from os import listdir
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from tensorflow.python.keras import models
from scipy.optimize import fmin_l_bfgs_b
import urllib
from tensorflow.python.keras.preprocessing import image as kp_image
import cv2
from IPython import display
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization, Dropout, Add

def style_transfer_test_data(img,width,height):
    img=cv2.resize(img, (width, height)) 
    test_image = np.array(img,dtype='float32')
    test_image = kp_image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    test_image /= 255
    # predictions
    test_img=transform_net(test_image,training=False)
    test_img=test_img.numpy()
    test_img=test_img[0,:,:,:]
    return np.clip(test_img,0,255).astype('uint8')

class TransformNet(tf.keras.Model):
    def __init__(self):
        super(TransformNet, self).__init__()
        # encode block
        self.block1_conv1 = Conv2D(filters=32, kernel_size=(9,9), strides=1, padding="same")
        self.block1_batchnorm1 = BatchNormalization(axis=3)
        self.block1_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same")
        self.block1_batchnorm2 = BatchNormalization(axis=3)
        self.block1_conv3 = Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same")
        self.block1_batchnorm3 = BatchNormalization(axis=3)
        # residual net block
        self.block2_conv1 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm1 = BatchNormalization(axis=3)
        self.block2_conv2 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm2 = BatchNormalization(axis=3)
        self.block2_add1 = Add()
        self.block2_conv3 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm3 = BatchNormalization(axis=3)
        self.block2_conv4 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm4 = BatchNormalization(axis=3)
        self.block2_add2 = Add()
        self.block2_conv5 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm5 = BatchNormalization(axis=3)
        self.block2_conv6 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm6 = BatchNormalization(axis=3)
        self.block2_add3 = Add()
        self.block2_conv7 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm7 = BatchNormalization(axis=3)
        self.block2_conv8 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm8 = BatchNormalization(axis=3)
        self.block2_add4 = Add()
        self.block2_conv9 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm9 = BatchNormalization(axis=3)
        self.block2_conv10 = Conv2D(filters=128, kernel_size=(3,3), padding="same")
        self.block2_batchnorm10 = BatchNormalization(axis=3)
        self.block2_add5 = Add()
        # decode block
        self.block3_conv1transpose = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same")
        self.block3_batchnorm1 = BatchNormalization(axis=3)
        self.block3_conv2transpose = Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding="same")
        self.block3_batchnorm2 = BatchNormalization(axis=3)
        self.block3_conv3transpose = Conv2D(filters=3, kernel_size=(9,9), strides=1, padding="same")
        self.block3_batchnorm3 = BatchNormalization(axis=3)
    def call(self, x, training=True):
        # encode block
        x = tf.reshape(x,(-1,512,512,3))
        x = self.block1_conv1(x)
        x = tf.nn.relu(x)
        x = self.block1_batchnorm1(x, training=training)
        x = self.block1_conv2(x)
        x = tf.nn.relu(x)
        x = self.block1_batchnorm2(x, training=training)
        x = self.block1_conv3(x)
        x = tf.nn.relu(x)
        x = self.block1_batchnorm3(x, training=training)

        # residual block
        x1 = x
        x = self.block2_conv1(x)
        x = self.block2_batchnorm1(x,training=training)
        x = tf.nn.relu(x)
        x = self.block2_conv2(x)
        x = self.block2_batchnorm2(x,training=training)
        x = self.block2_add1([x, x1])
        x1 = x
        x = self.block2_conv3(x)
        x = self.block2_batchnorm3(x,training=training)
        x = tf.nn.relu(x)
        x = self.block2_conv4(x)
        x = self.block2_batchnorm4(x,training=training)
        x = self.block2_add2([x, x1])
        x1 = x
        x = self.block2_conv5(x)
        x = self.block2_batchnorm5(x,training=training)
        x = tf.nn.relu(x)
        x = self.block2_conv6(x)
        x = self.block2_batchnorm6(x,training=training)
        x = self.block2_add3([x, x1])
        x1 = x
        x = self.block2_conv7(x)
        x = self.block2_batchnorm7(x,training=training)
        x = tf.nn.relu(x)
        x = self.block2_conv8(x)
        x = self.block2_batchnorm8(x,training=training)
        x = self.block2_add4([x, x1])
        x1 = x
        x = self.block2_conv9(x)
        x = self.block2_batchnorm9(x,training=training)
        x = tf.nn.relu(x)
        x = self.block2_conv10(x)
        x = self.block2_batchnorm10(x,training=training)
        x = self.block2_add5([x, x1])

        # decode block
        x = self.block3_conv1transpose(x)
        x = tf.nn.relu(x)
        x = self.block3_batchnorm1(x,training=training)
        x = self.block3_conv2transpose(x)
        x = tf.nn.relu(x)
        x = self.block3_batchnorm2(x,training=training)
        x = self.block3_conv3transpose(x)
        x = (tf.nn.tanh(x)+1)*127.5  
        #so as to input vgg16, this idea comes from 
        # https://github.com/malhagi/tensorflow-fast-neuralstyle
        return x
    
def load_model():
    transform_net=TransformNet()
    # Initialize somehow!
    _=transform_net(tfe.Variable(np.zeros((1,512,512,3),dtype=np.float32)), training=True)
    web='http://www.columbia.edu/~hl3099/aml/beta_model_style_1.h5' 
    urllib.request.urlretrieve(web,'beta_model_style_1.h5')
    transform_net.load_weights('beta_model_style_1.h5',by_name=False)
    return transform_net
  
transform_net=load_model()

layout = html.Div([ 
    nav_bar,
    upload_img
])