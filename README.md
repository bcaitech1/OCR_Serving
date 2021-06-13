## OCR Serving


#### Stack
- Flask
- torch


### model weight 
학습된 모델을 다운 링크 입니다. Model 디렉토리 안에 넣고 모델 불러오는 함수 호출 할때 인자로 해당 파일 경로를 넣어야 합니다.
- SATRN : [Download](https://drive.google.com/file/d/1HhDG3eNpVO4VICIGvZzBsdTbRFS8mODA/view?usp=sharing)

### Docker build & Run
```angular2html
# build
docker bulid -t unnamed_OCR . 
# run
docker run --gpus all -it -p 8000:5000 unnamed_OCR
python3 p4_ocr.py
```


### model 사용 예시

```
from Model.web_inference import OCRModel

# model initialize
model = OCRmodel( token_path = token_path,
            model_path = model_path
         )
# model load
model.load()

# inference
sequence_str, latency = model.inference( image = image )
# iamge = numpy.arr 형태의 이미지 vector (cv2로 불러온)
```