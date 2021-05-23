from flask import Flask,render_template,request,send_from_directory
import os
import keras
from keras.models import load_model
#import tensorflow
import cv2
import numpy as np
import pandas



app = Flask(__name__)

STATIC_FOLDER = 'stati'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploadd'
#FOLDER = 'Gallery'
#ROH_FOLDER = FOLDER + '/galleryuploads'



model=load_model('cnn_.h5')

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods = ['POST'])
def predict():
    img = request.files['query']
    fullname = os.path.join(UPLOAD_FOLDER, img.filename)
    img.save(fullname)
    img_arr = cv2.imread(fullname)
    img_arr = cv2.resize(img_arr, (64, 64))
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    # scaling to 0 to 1
    if (np.max(img_arr) > 1):
        img_arr = img_arr / 255.0
    img_arr = np.array([img_arr])
    prediction = model.predict_classes(img_arr)
    label = ['cat', 'dog']
    prediction = prediction[0][0]
    pree = label[prediction]
    #if pree == 'cat':
        #pree = os.path.join()
    return render_template('predict.html',var = pree)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)




if __name__ == '__main__':
    app.run(debug=True)


