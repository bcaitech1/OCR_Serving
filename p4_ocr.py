
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from Model.web_inference import OCRModel

#test
# joonam test

UPLOAD_FOLDER = "C:\\Users\\l_jad\\Desktop\\OCR_serving\\static\\files"
LATEX_STRING = " "
FILE_INFO = ""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model load
model = OCRModel( token_path = "./Model/gt.txt" ,
                  model_path = "./Model/weights.pth")
model.load()

@app.route('/', methods=['GET', 'POST'])
def mainPage():
    global LATEX_STRING, FILE_INFO

    if request.method == 'POST':
        file = request.files['img'] # werkzeug.datastructrues.FileStorage
        if file:
            user_file_name = secure_filename(file.filename)
            FILE_INFO = user_file_name
            print(user_file_name)


            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            ## model run
            sequence_str, latency = model.inference(image = img)
            print(latency) # 소요시간.
            ##

            '''
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "user_file.png")) # js 쓰면 파일 저장 안 하고도 전달 가능

            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], "user_file.png"), cv2.IMREAD_COLOR)
            '''
            std_img = round(img.std(), 3)
            mean_img = round(img.mean(), 3)
            
            FILE_INFO = "std : " + str(std_img) + ", mean : " + str(mean_img) # 서버 단에서 입력된 이미지를 처리해 돌려주는 게 가능하다는 것을 증명

            # model output
            LATEX_STRING = sequence_str
            LATEX_STRING = "$$s^a+m^p+l^e=\\frac{equa}{tion}$$"
            #return jsonify({'LATEX':LATEX_STRING}) # js를 사용해 보자

    return render_template('main.html', INFO=FILE_INFO, LATEX=LATEX_STRING)
    # render_template는 새로고침해서 '/' 경로로 요청이 와야 변수를 날릴 수 있나 봐...
    # ajax 써서 비동기로 날렸으니 이 변수들 전달이 안 됨


if __name__ == "__main__":
    app.run(host='localhost', debug=True, port=80)







