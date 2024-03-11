from ..resnet.resnet import ResNet50
import torch
from torchvision import transforms
from PIL import Image
import sys


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 训练方法
model = ResNet50()

log_dir = "C:/Users/z/Desktop/interpretability/dev/run/resnet_50.pth"

path = "C:/Users/z/Desktop/interpretability/datasets/val/n01518878/ILSVRC2012_val_00048600.JPEG"

def testOnce(path) :
    image = Image.open(path)
    image = test_transform(image)
    image = torch.reshape(image, (1, 3, 224, 224))

    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available() :
        model.cuda()

    model.eval()
    with torch.no_grad() :
        image = image.cuda()
        out = model(image)
        _, predicted = torch.max(out, 1)
        print(predicted)
    

if __name__ == '__main__' :
    path = sys.argv[1] or path

    print(sys.argv[1])

    testOnce(path)