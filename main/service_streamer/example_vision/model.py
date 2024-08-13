# coding=utf-8
# Created by Meteorix at 2019/8/9
import io
import json
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import os

pwd = os.path.abspath(os.path.dirname(__file__))
index_path = os.path.join(pwd, 'imagenet_class_index.json')

imagenet_class_index = json.load(open(index_path))
device = "cuda"
# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
model.to(device)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes).to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


def batch_prediction(image_bytes_batch):
    image_tensors = [transform_image(image_bytes=image_bytes) for image_bytes in image_bytes_batch]
    tensor = torch.cat(image_tensors).to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_ids = y_hat.tolist()
    return [imagenet_class_index[str(i)] for i in predicted_ids]

def get_probality(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes).to(device)
        outputs = torch.softmax(model.forward(tensor).squeeze(),dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(imagenet_class_index[str(index)],float(p)) for index, p in enumerate(prediction)]
        #sort probality
        index_pre.sort(key=lambda x:x[1],reverse = True)
        text = [template.format(k[1],v) for k,v in index_pre]
        return_info =  {'result':text}
    except Exception as e:
        return_info =  {"result": [str(e)]}
    return return_info

#需要先更改densenet.py 将forward重写为extratror
def get_vector(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes).to(device)
        outputs = model(tensor)
        outputs = torch.softmax(outputs.squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        return_info =  {'result':prediction.tolist()}
    except Exception as e:
        return_info =  {"result": [str(e)]}
    return return_info


if __name__ == "__main__":
    with open(r"./example_vision/cat.jpg", 'rb') as f:
        image_bytes = f.read()

    result = get_vector(image_bytes)
    print(result)
    # batch_result = batch_prediction([image_bytes])
    # assert batch_result == [result] * 64
    # print(batch_result)
