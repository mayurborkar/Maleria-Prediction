
from __future__ import division, print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# Coding UTF-8
import sys 
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask Utlisis
from flask import Flask,render_template,request,url_for,redirect
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Create the app
app = Flask(__name__)

# Model save with keras .save()
model_path = 'model_vgg19.h5'

# Load The Save Model
model = load_model(model_path)


def model_predict(img_path,model):
    img = image.load_model(img_path,target_size=(224,224))
    
    #Preprocess The Image
    x=image.img_to_array(img)
    # x = np.true_divide(x,255)
    ## Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)
    
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With Pneumonia"
    else:
        preds="The Person is not Infected With Pneumonia"
    
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main Page
    return render_template('index.html')

@app.route('/ predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get The File From Post Request
        f = request.files['file']
        
        # Save The File ./Upload
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make Prediction
        preds = model_predict(file_path,model)
        result = preds
        return result
    return None

if __name__ == '__main':
    app.run(port=5001, debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    