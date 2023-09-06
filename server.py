import json
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import cv2 as cv2
import matplotlib.pyplot as plt
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

classes = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
model = load_model('model.h5')
def return_prediction(model, img):
    print ('Input image shape is ', img.shape)
    # img_size=(300,300)
    img_size=(180,180)
    img=cv2.resize(img, img_size)
    print ('the resized image has shape ', img.shape)
    img=np.expand_dims(img, axis=0)
    # print ('image shape after expanding dimensions is ',img.shape)
    pred=model.predict(img)
    print ('the shape of prediction is ', pred.shape)
    # index=np.argmax(pred[0])
    index=np.argmax(pred)
    klass=classes[index]
    # probability=pred[0][index]*100
    print(f'the image is predicted as being {klass} %')
    res = [klass]
    return res

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/pred', methods=['POST'])
def pred():
    img = plt.imread(request.files['myImage']) #flask.request.files('imagefile', '')
    result =  return_prediction(model, img)
    print(f"results : {result} ")
    return json.dumps({'cat': result[0]})
    # return render_template('index.html', result= result)


if __name__=='__main__':
    app.run()