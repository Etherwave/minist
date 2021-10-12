# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F #加载nn中的功能函数
import torch.optim as optim #加载优化器有关包
import torch.utils.data as Data
from torchvision import datasets,transforms #加载计算机视觉有关包
from torch.autograd import Variable
import os
import cv2
import numpy as np
from build_model import perfect_build_moudel
from build_model import Layer
from Minist_Dataset import MinistDataset

BATCH_SIZE = 64

model_state_dict_path = "./save/model.state_dict"

model_path = "./save/model.model"

minist_dataset = MinistDataset()

#定义网络模型

class Conv_kernal(nn.Module):
    def __init__(self, in_channl, out_channl, conv_kernel_size, conv_stride, pool_kernel_size, pool_stride):
        super(Conv_kernal, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channl, out_channl, kernel_size=(conv_kernel_size, conv_kernel_size),
                      stride=(conv_stride, conv_stride)),
            nn.BatchNorm2d(out_channl),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        return self.conv(X)

class Model(nn.Module):
    def __init__(self, image_size, output_size):
        super(Model,self).__init__()
        self.image_size = image_size
        self.output_size = output_size
        block_configs = [1, 5, 10]

        self.features = nn.Sequential()

        layers = perfect_build_moudel(block_configs, self.image_size, self.output_size)

        for i in range(len(block_configs)-1):
            self.features.add_module(str(i), Conv_kernal(block_configs[i], block_configs[i+1],
                layers[i*2].kernel_size, layers[i*2].stride, layers[i*2+1].kernel_size, layers[i*2+1].stride))

        self.features.add_module("bn", nn.BatchNorm2d(block_configs[-1]))

    def forward(self, image):
        return self.features(image)

def save_model_state_dict(model, model_state_dict_path):
    if os.path.exists("./save")==False:
        os.mkdir("./save")
    try:
        torch.save(model.state_dict(), model_state_dict_path)
        print("model state_dict save success!")
    except:
        print("model state_dict save failed!")

def load_model_state_dict(model, model_state_dict_path):
    try:
        model.load_state_dict(torch.load(model_state_dict_path))
        print("model state_dict load success!")
    except:
        print("model state_dict load failed!")

def save_model(model, model_path):
    if os.path.exists("./save")==False:
        os.mkdir("./save")
    try:
        torch.save(model, model_path)
        print("model save success!")
    except:
        print("model save failed!")

def load_model(model_path):
    model = None
    try:
        model = torch.load(model_path)
        print("model load success!")
    except:
        print("model load failed!")
    return model

def train():
    BATCH_SIZE = 64

    # 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
    data_loader = Data.DataLoader(dataset=minist_dataset, batch_size=BATCH_SIZE, shuffle=False)

    image_size = 28
    output_size = 1

    model = Model(image_size, output_size)
    load_model_state_dict(model, model_state_dict_path)
    model.train()
    loss = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 2

    for echo in range(num_epochs):
        train_loss = 0   #定义训练损失
        train_acc = 0    #定义训练准确度
        model.train()    #将网络转化为训练模式
        for index, (images, labels) in enumerate(data_loader):     #使用枚举函数遍历train_loader
            # images torch.Size([64, 1, 28, 28])
            images = images.float()
            labels = labels.long()
            out = model(images)           #正向传播
            # out.shape torch.Size([64, 10, 1, 1])
            out = out.reshape([out.shape[0], -1])
            # out.shape torch.Size([64, 10])
            lossvalue = loss(out, labels)         #求损失值
            optimizer.zero_grad()       #优化器梯度归零
            lossvalue.backward()    #反向转播，刷新梯度值
            optimizer.step()        #优化器运行一步，注意optimizer搜集的是model的参数

            #计算损失
            train_loss += float(lossvalue)
            #计算精确度
            _,pred = out.max(1)
            num_correct = (pred == labels).sum()
            acc = int(num_correct) / images.shape[0]
            train_acc += acc
        print("echo:"+' ' +str(echo))
        print("lose:" + ' ' + str(train_loss / data_loader.__len__()))
        print("accuracy:" + ' '+str(train_acc / data_loader.__len__()))
    save_model_state_dict(model, model_state_dict_path)

    save_model(model, model_path)


def test():
    model = load_model(model_path)
    model.eval()

    for i in range(minist_dataset.size):
        image = minist_dataset.test_images[i]
        label = minist_dataset.test_labels[i]
        # print(np.shape(image))
        # (28, 28)
        image_add_dim = np.expand_dims(image, axis=0)
        image_add_dim = np.expand_dims(image_add_dim, axis=0)
        image_tensor = torch.Tensor(image_add_dim)
        output = model(image_tensor)
        output = output.reshape([output.shape[0], -1])
        index = torch.argmax(output)
        print("predict : {0}, gt : {1}".format(index, label))

        cv2.imshow("1", image)
        cv2.waitKey()
    cv2.waitKey()




if __name__ == '__main__':
    # train()
    test()
