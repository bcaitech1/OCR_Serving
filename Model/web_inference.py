

# 이미지 형식은 리스트로 받는걸로,
# 모델을 실행하여 서버 메모리를 점유 ( class.load() )
# 받은 이미지를 inference 한 후에 실행하는 방식

import torch
from torchvision import transforms
from PIL import Image
import time
import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def load_vocab(tokens_paths):
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    with open(tokens_paths, "r") as fd:
        reader = fd.read()
        for token in reader.split("\n"):
            if token not in tokens:
                tokens.append(token)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token

def id_to_string(tokens, vocab ,do_eval=0):
    result = []
    if do_eval:
        special_ids = [vocab["token_to_id"]["<PAD>"], vocab["token_to_id"]["<SOS>"],
                       vocab["token_to_id"]["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += vocab["id_to_token"][token] + " "
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += vocab["id_to_token"][token] + " "

        result.append(string)
    return result


class OCRModel():
    def __init__(self, token_path, model_path ):
        token_to_id, id_to_token = load_vocab(token_path)
        self.vocab = dict(
            token_to_id = token_to_id,
            id_to_token = id_to_token
                )
        self.model_path = model_path
        # self.transformed = A.Compose(
        # [
        #     A.Normalize(always_apply=True),
        #     ToTensorV2()
        # ]
        # )
        self.transformed = transforms.Compose(
            [
                transforms.Resize((128,128)),
                transforms.ToTensor(),

            ]
        )

    def load(self):
        self.model = torch.load(self.model_path)
        print('model load')


    def run_model(self, input_image, dummy_gt):
        start = time.time()

        self.model.to('cuda')
        self.model.eval()
        #####
        # output = model(sample['image'].to('cuda'), sample["truth"]["encoded"].to('cuda'), False, 0.0)
        output = self.model(input_image.to('cuda'), dummy_gt.to('cuda'), False, 0.0)
        ####
        decoded_values = output.transpose(1, 2)
        _, sequence = torch.topk(decoded_values, 1, dim=1)
        sequence = sequence.squeeze(1)
        sequence_str = id_to_string(sequence, self.vocab, do_eval=1)

        # print(sequence_str)
        # print(f"latency : {time.time() - start}")
        return sequence_str, time.time() - start


    def inference(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128,128))
        image = image/255
        test = torch.tensor(image, dtype=torch.float)
        test = test.unsqueeze(0)
        # test = self.transformed(image=image)
        # test = test["image"]
        test = torch.stack([test, test], dim=0)
        dummy_gt = torch.zeros((2, 232)) + 158
        sequence_str, latency = self.run_model(test, dummy_gt)
        return sequence_str, latency


    def inference_(self, image_path):
        rgb = 1
        input_image = Image.open(image_path)
        if rgb == 3:
            input_image = input_image.convert("RGB")
        elif rgb == 1:
            input_image = input_image.convert("L")
        test = self.transformed(input_image)
        test = torch.stack([test,test], dim=0)
        dummy_gt = torch.zeros((2,232))+158
        sequence_str, latency = self.run_model(test, dummy_gt)
        return sequence_str, latency


