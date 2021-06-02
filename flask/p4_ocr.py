
import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "C:\\Users\\l_jad\\Desktop\\OCR_serving\\static\\files"
LATEX_STRING = "DUMMY"
FILE_INFO = ""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def mainPage():
    global LATEX_STRING, FILE_INFO

    if request.method == 'POST':
        file = request.files['img']
        if file:
            #user_file = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "user_file.png")) # js 쓰면 파일 저장 안 하고도 전달 가능

            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], "user_file.png"), cv2.IMREAD_COLOR)
            std_img = round(img.std(), 3)
            mean_img = round(img.mean(), 3)
            
            FILE_INFO = "std : " + str(std_img) + ", mean : " + str(mean_img) # 서버 단에서 입력된 이미지를 처리해 돌려주는 게 가능하다는 것을 증명
            
            LATEX_STRING = "$$s^a+m^p+l^e=\\frac{equa}{tion}$$"
            #return jsonify({'LATEX':LATEX_STRING}) # js를 사용해 보자

    return render_template('_main.html', INFO=FILE_INFO, LaTeX=LATEX_STRING)


if __name__ == "__main__":
    app.run(host='localhost', debug=True, port=80)







