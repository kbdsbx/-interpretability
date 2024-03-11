from ..resnet.resnet import ResNet50
import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
import os
import time
import matplotlib.pyplot as plt
from PIL import Image


# model = ResNet101()
# print(model)
# input = torch.randn(1, 3, 224, 224)
# out = model(input)
# print(out.shape)

# 训练参数
epochs = 150
batch_size = 64
classes_num = 10
learning_rate = 1e-3
momentum = 0.9
weight_decay= 1e-4


# 对数据集进行处理
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224), # 随机剪裁
    # transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 使用均值和方差（ILSVRC2012）进行归一化
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 数据集地址
train_dir = "C:/Users/z/Desktop/interpretability/datasets/train"
train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_dir = "C:/Users/z/Desktop/interpretability/datasets/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transform)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


# 训练方法
model = ResNet50()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum= momentum, weight_decay= weight_decay)
# 学习率动态调度
stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_func = torch.nn.CrossEntropyLoss()

# 暂存
loss_list = []
accuracy_list = []
train_loss_list = []
train_accuracy_list = []


def train_res (model, train_dataloader) :
    model.train()

    train_loss = 0.
    train_acc = 0.
    idx = 1
    
    for batch_x, batch_y in train_dataloader :
        # 送入显存
        batch_x = Variable(batch_x).cuda()
        batch_y = Variable(batch_y).cuda()

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        out = model(batch_x)

        # 损失计算
        loss = loss_func(out, batch_y)

        train_loss += loss.item()

        # 获取预测值
        pred = torch.max(out, 1)[1]

        # 计算准确率
        train_corrent = (pred == batch_y).sum()
        train_acc += train_corrent.item()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()
        print("Training batch: {}".format(idx))
        idx+=1
    
    print("Train Loss: {:.6f}, Acc: {:.6f}".format(train_loss / len(train_datasets), train_acc / len(train_datasets)))
    train_loss_list.append(train_loss / len(train_datasets))
    train_accuracy_list.append(train_acc / len(train_datasets) * 100)

def val(model, val_dataloader) :
    model.eval()

    val_loss = 0.
    val_acc = 0.

    with torch.no_grad() :
        for batch_x, batch_y in val_dataloader :
            batch_x = Variable(batch_x).cuda()
            batch_y = Variable(batch_y).cuda()

            out = model(batch_x)
            
            # 损失计算
            loss = loss_func(out, batch_y)

            val_loss += loss.item()

            # 获取预测值
            pred = torch.max(out, 1)[1]

            # 计算准确率
            val_corrent = (pred == batch_y).sum()
            val_acc += val_corrent.item()
    
    print("Validation Loss: {:.6f}, Acc: {:.6f}".format(val_loss / len(val_datasets), val_acc / len(val_datasets)))
    loss_list.append(val_loss / len(val_datasets))
    accuracy_list.append(val_acc / len(val_datasets) * 100)


log_dir = "C:/Users/z/Desktop/interpretability/dev/run/resnet_50.pth"

def testing () :
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available() :
        model.cuda()

    val(model, val_dataloader)

def training () :
    if torch.cuda.is_available() :
        model.cuda()
    
    if os.path.exists(log_dir) :
        checkpoint = torch.load(log_dir)
        # 从检查点加载
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else :
        start_epoch = 1


    for epoch in range(start_epoch, epochs) :
        tic = time.time()
        print('epoch {}'.format(epoch))
        train_res(model, train_dataloader)
        toc = time.time()

        print("Training complete in {:.0f}m {:.0f}s". format((toc - tic) // 60, (toc-tic) % 60))

        print("evaluating model...")
        val(model, val_dataloader)
        state = {
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch' : epoch
        }
        torch.save(state, log_dir)
        

def printing() :
    plt.subplot(2, 1, 1)
    plt.plot(range(len(accuracy_list)), accuracy_list, '-')
    plt.xlabel('Validation accuracy vs. epoches.')
    plt.ylabel('Validation accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('Validation accuracy vs. epoches.')
    plt.ylabel('Validation loss')
    plt.show()


if __name__ == '__main__' :
    # path = "C:/Users/z/Desktop/interpretability/datasets/val/n01440764/ILSVRC2012_val_00009396.JPEG"
    # path = "C:/Users/z/Desktop/interpretability/datasets/val/n01498041/ILSVRC2012_val_00001935.JPEG"
    # path = "C:/Users/z/Desktop/interpretability/datasets/val/n01518878/ILSVRC2012_val_00037941.JPEG"
    # testOnce(path)
    testing()
    # print(accuracy_list)
    # print(loss_list)
    # printing()