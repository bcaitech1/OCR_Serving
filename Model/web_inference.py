

# 이미지 형식은 리스트로 받는걸로,
# 모델을 실행하여 서버 메모리를 점유 ( class.load() )
# 받은 이미지를 inference 한 후에 실행하는 방식


class Inference():
    def __init__(self):
        self.checkpoint_path