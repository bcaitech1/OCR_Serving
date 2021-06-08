## OCR Serving


#### Stack
- Flask
- torch


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