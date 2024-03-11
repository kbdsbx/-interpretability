import torch
import numpy as np


# cam can only be used in model which with a avgpool layer.

def grad_cam(model, layer, sample) :
    """
    model : 模型
    layer : 绘制gcam图的层
    sample : 处理好的样本，通常格式为(1, 3, w, h)形状的张量
    """

    if torch.cuda.is_available() :
        model.cuda()

    model.eval()

    features = []
    weights = []

    def feature_hooks(module, input, output) :
        features.append(output.data.cpu().numpy())
    
    def weight_hooks(moudle, grad_in, grad_out) :
        ws = torch.sum(grad_out[0].detach(), [2, 3])
        weights.append(ws.data.cpu().numpy())

    # 获取正向传播的特征图
    layer.register_forward_hook(feature_hooks)
    # 获取反向传播的梯度
    layer.register_backward_hook(weight_hooks)

    sample = sample.cuda()

    out = model(sample)
    _, predicted = torch.max(out, 1)
    # 预测类别
    pred_idx = predicted[0]

    # grad-cam 由最高评分位置反向传播，其余位置不参与传播
    model.zero_grad()
    score = out[0, pred_idx]
    score.requires_grad_(True)
    score.backward()

    feature = features[0]
    weight = weights[0]

    # 计算显著度
    _, c, h, w = feature.shape
    gcam = np.dot(weight, feature.reshape((c, h * w))).reshape(h, w)

    # ReLU
    gcam = np.maximum(0, gcam)

    # 归一化
    gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min())
    gcam = np.uint8(255 * gcam)

    return gcam
