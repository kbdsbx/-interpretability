import torch
from ..resnet.resnet import ResNet50
from ..explainability.grad_cam import grad_cam
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

import sys

log_dir = "C:/Users/z/Desktop/interpretability/dev/run/resnet_50.pth"
model = ResNet50()

path = "C:/Users/z/Desktop/interpretability/datasets/web-img/0f499463cbc447af9b101129b3e449b9.jpg"


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if __name__ == "__main__" :
    if len(sys.argv) > 1 :
        path = sys.argv[1]
    print(path)

    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['model'])

    # 打印所有层
    # for n,v in model.named_parameters():
        # print(n)
        # print(v)
    
    img = Image.open(path)
    sample = test_transform(img)
    sample = torch.reshape(sample, (1, 3, 224, 224))

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    for n,v in model.named_modules() :
        if isinstance(v, torch.nn.Conv2d) or isinstance(v, torch.nn.BatchNorm2d) :
            cam_map = grad_cam(model, v, sample)
            heatmap = cv2.applyColorMap(cv2.resize(cam_map, (w, h)), cv2.COLORMAP_JET)
            result = heatmap * 0.4 + img * 0.6
            cv2.imwrite('output/grad_cam/gcam{}.jpg'.format(n), result)

    
    # cam_map = grad_cam(model, model.layer4[2].bottleneck[7], path)
    # print(cam_map)

    # img = cv2.imread(path)
    # h, w, _ = img.shape
    # heatmap = cv2.applyColorMap(cv2.resize(cam_map, (w, h)), cv2.COLORMAP_JET)
    # result = heatmap * 0.4 + img * 0.6
    # cv2.imwrite('gcam.jpg', result)
