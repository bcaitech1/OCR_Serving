
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from Model.web_inference import OCRModel


## model load
model = OCRModel( token_path = "./Model/tokens.txt" ,
                  model_path = "./Model/test_model.pth")
model.load()
##

## initialize flask app
app = Flask(__name__)
LATEX_STRING = "$$s^a+m^p+l^e=\\frac{equa}{tion}$$"
##

@app.route('/', methods=['GET', 'POST'])
def mainPage():
    global LATEX_STRING

    if request.method == 'POST':
        file = request.files['img'] # type : werkzeug.datastructrues.FileStorage
        if file:
            ## read uploaded image
            file_str = file.read()
            img_np = np.frombuffer(file_str, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR) # type : np.ndarray
            ##

            ## model run
            sequence_str, latency = model.inference(image = img)
            print(latency) # 소요 시간
            print(sequence_str[0])

            LATEX_STRING = "$$"+sequence_str[0]+"$$"

            return jsonify({'result': LATEX_STRING}) # ajax를 이용해 비동기적으로 응답 전달

    return render_template('main.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)







