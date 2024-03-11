import torch
import numpy as np


# cam can only be used in model which with a avgpool layer.

def cam(model, layer, fc, sample) :
    """
    model : pytorch模型
    layer : 平均池化前的一层
    fc : 全连接层，平均池化的后一层
    sample : 处理好的样本，通常格式为(1, 3, w, h)形状的张量
    """
    
    if torch.cuda.is_available() :
        model.cuda()

    model.eval()

    # 获取特征图
    features = []

    def hooks(module, input, output) :
        features.append(output.data.cpu().numpy())

    # 给某一层添加钩子，获取输入/输出
    layer.register_forward_hook(hooks)

    with torch.no_grad() :
        sample = sample.cuda()

        out = model(sample)
        _, predicted = torch.max(out, 1)
        # 预测类别
        pred_idx = predicted[0]

        # 前向传播特征图
        feature = features[0]

        weights = fc.weight.data.cpu().numpy()
        # 对应预测类别的权重
        weight = weights[pred_idx]

        # 计算显著度
        _, c, h, w = feature.shape
        cam = np.dot(weight, feature.reshape((c, h * w))).reshape(h, w)

        # 归一化
        cam = (cam - cam.min()) /(cam.max() - cam.min())
        cam = np.uint8(255 * cam)

        return cam
    



