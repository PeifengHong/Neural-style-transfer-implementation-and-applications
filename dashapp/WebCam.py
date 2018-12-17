import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from App import app
from App import server
from flask import Flask, Response
import cv2
#Eager Mode
import tensorflow as tf
tf.enable_eager_execution()    
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
from IPython import display
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization, Dropout,Add
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
def load_model(style_select):
    transform_net=TransformNet()
    # Initialize somehow!
    _=transform_net(tfe.Variable(np.zeros((1,512,512,3),dtype=np.float32)), training=True)
    if style_select == 'style1':
        web='http://www.columbia.edu/~hl3099/aml/beta_model_style_1.h5'
        urllib.request.urlretrieve(web,'beta_model_style_1.h5')
        transform_net.load_weights('beta_model_style_1.h5',by_name=False)
    elif style_select == 'style2':
        web='http://www.columbia.edu/~hl3099/aml/beta_model_style_2.h5'
        urllib.request.urlretrieve(web,'beta_model_style_2.h5')
        transform_net.load_weights('beta_model_style_2.h5',by_name=False)
    else:
        web='http://www.columbia.edu/~hl3099/aml/beta_model_style_3.h5'
        urllib.request.urlretrieve(web,'beta_model_style_3.h5')
        transform_net.load_weights('beta_model_style_3.h5',by_name=False)
    return transform_net
	
def style_transfer_test_data(img,width,height,transform_net):
  
    # read data
    #test_image = Image.open(i)
    img=cv2.resize(img, (width, height)) 
    test_image = np.array(img,dtype='float32')
    test_image = kp_image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    test_image /= 255
    #test_images[0,:,:,:]=np.asarray(test_image, dtype='float32')
  
    # predictions
    test_img=transform_net(test_image,training=False)
    test_img=test_img.numpy()
    test_img=test_img[0,:,:,:]
    return np.clip(test_img,0,255).astype('uint8')  

@app.callback(Output('show-style', 'children'),
			 [Input('style-select','value'),]
              )
def update_style(style_select):
    if style_select == 'Starry Night Over the Rhône':
        transform_net = transform_net1
    elif style_select == 'Victoire':
        transform_net = transform_net2
    else:
        transform_net = transform_net3
    return html.Div(children=[html.H3(style_select + ' style camera'),
                        html.Img(src=app.get_asset_url(style_select + '.jpg'), width="25%"),
])
    

@app.callback(Output('Webcam-Display', 'children'),
              [Input('Start-webcam', 'n_clicks'),			  
             ],
			 [State('style-select','value'),]
              )
def start_webcam(start_click,style_select):
    if style_select == 'Starry Night Over the Rhône':
        transform_net = transform_net1
    elif style_select == 'Victoire':
        transform_net = transform_net2
    else:
        transform_net = transform_net3
    video_capture = cv2.VideoCapture(0)
    count = 0
    while start_click>0:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Draw a rectangle around the faces
        #for (x, y, w, h) in faces:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        frame=style_transfer_test_data(frame,512,512,transform_net)
        
        # Display the resulting frame
        cv2.imshow('Video', frame[:,:,::-1])
       
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
	
# Configure navbar menu
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



transform_net1=load_model('style1')
transform_net2=load_model('style2')
transform_net3=load_model('style3')

all_options = ['Starry Night Over the Rhône', 'Victoire', 'Women at Their Toilette']

layout = html.Div([
    nav_bar,
    html.Br(),
    html.H1('Enjoy WebCam Time'),
    html.Br(),
    html.Div([
        html.Img(src=app.get_asset_url('Starry Night Over the Rhône.jpg'), width="33%", height="300px"),
        html.Img(src=app.get_asset_url('Victoire.jpg'), width="33%", height="300px"),
        html.Img(src=app.get_asset_url('Women at Their Toilette.jpg'), width="33%", height="300px")
    ]),
    dcc.RadioItems(
        id='style-select',
        options=[{'label': k, 'value': k} for k in all_options],
        value='Starry Night Over the Rhône',
        style={"padding":"10px"}
    ),
    html.Div(id='Webcam-Display'),
    html.Div(id='show-style'),
    html.Br(),
    html.Button(id='Start-webcam',n_clicks=0, children='Start', style={'textAlign': 'center','height':'30px','width':'20%','backgroundcolor':'#F9E79F'}),
    html.H3('Press Q to quit.')   
], className="jumbotron text-center")