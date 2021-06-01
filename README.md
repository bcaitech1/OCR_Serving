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
sequence_str, latency = model.inference( image_path = image_path )
```