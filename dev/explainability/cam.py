from PIL import Image
from torch.autograd import Variable


# cam can only be used in model which with a avgpool layer.


def cam(model, layer_name, path, transform) :
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    x = Variable(img).reshape(1, 3, 224, 224)
    
    # 获取所有特征图
    features_blobs = []

    model._modules.get(layer_name).register_forward_hood(lambda module, input, output : {
        features_blobs.append(output.data.cpu().numpy())
    })


