import torch
from ..resnet.resnet import ResNet50
from ..explainability.cam import cam
from torchvision import transforms

log_dir = "C:/Users/z/Desktop/interpretability/dev/run/resnet_50.pth"
model = ResNet50()


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

path = "C:/Users/z/Desktop/interpretability/datasets/val/n01518878/ILSVRC2012_val_00037941.JPEG"

if __name__ == "__main__" :
    checkpoint = torch.load(log_dir)
    model.eval()
    model.load_state_dict(checkpoint['model'])
    if torch.cuda.is_available() :
        model.cuda()

    # 打印所有层的名字
    # for n,v in model.named_parameters():
        # print(n)
    
    cam(model, 'layer4.2.bottleneck.7.weight', path, test_transform)
