import torch
import numpy as np

class Layer:

    def __init__(self, kernel_size=0, stride=0):
        self.kernel_size = kernel_size
        self.stride = stride

    def __str__(self):
        return "%d,%d" % (self.kernel_size, self.stride)



def calc_output_size(n, p, f, s):
    return int((n+2*p-f)/s+1)

def imperfect_build_moudel(block_config=[], image_size=4, output_size=1, min_kernel_size=2, max_kernel_size=10, min_stride=1, max_stride=5):
    '''
    assume every block is looks like follows
    self.conv = nn.Sequential(
                    nn.Conv2d(in_channl, out_channl, kernel_size=(9, 9), stride=(2, 2)),
                    nn.BatchNorm2d(out_channl),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.ReLU(inplace=True),
                )
    此函数容易找到解，但是每一层的kernel_size和stride的设置并不保证不丢失特征
    比如有可能在featrue_size=4的时候使用kernel_size=3,stride=2使得输出的feature_size=1
    此函数会输出符合要求的最小的kernel_size和stride
    :param config: 期望的每个block的channel
    :param image_size: 输入的图片大小
    :param output_size: 期望的输出的特征大小
    :param min_kernel_size: 期望的最小的kernel_size
    :param max_kernel_size: 期望的最大的kernel_size
    :param min_stride: 期望的最小的stride
    :param max_stride: 期望的最大的stride
    :return: 符合要求的一个layer配置解
    '''

    mmap_size = image_size
    layer_number = (len(block_config)-1)*2
    mmap = np.zeros((layer_number+1, mmap_size+1)).astype(int)
    mmap[0][image_size] = 1
    pre = np.zeros((layer_number + 1, mmap_size + 1)).astype(int)
    layers = [[] for i in range(layer_number+1)]
    for i in range(len(layers)):
        for j in range(mmap_size+1):
            layers[i].append(Layer())

    for i in range(layer_number):
        for j in range(mmap_size+1):
            if mmap[i][j] == 1:
                for f in range(max_kernel_size, min_kernel_size-1, -1):
                    for s in range(max_stride, min_stride-1, -1):
                        t_output_size = calc_output_size(j, 0, f, s)
                        if t_output_size >= output_size and t_output_size <= image_size:
                            mmap[i+1][t_output_size] = 1
                            layers[i+1][t_output_size] = Layer(f, s)
                            pre[i+1][t_output_size] = j

    ans_layers = []
    if mmap[layer_number][output_size] == 1:
        pre_no = output_size
        for i in range(layer_number, 0, -1):
            ans_layers.append(layers[i][pre_no])
            pre_no = pre[i][pre_no]

    ans_layers.reverse()

    return ans_layers

def perfect_build_moudel(block_config=[], image_size=4, output_size=1, min_kernel_size=2, max_kernel_size=3, min_stride=1, max_stride=2):
    '''
    assume every block is looks like follows
    self.conv = nn.Sequential(
                    nn.Conv2d(in_channl, out_channl, kernel_size=(9, 9), stride=(2, 2)),
                    nn.BatchNorm2d(out_channl),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.ReLU(inplace=True),
                )
    此函数不容易找到解，但是每一层的kernel_size和stride的设置可以保证不丢失特征
    比如不可能在featrue_size=4的时候使用kernel_size=3,stride=2使得输出的feature_size=1
    只会在featrue_size=4的时候使用kernel_size=4,stride=1使得输出的feature_size=1，保证了特征不丢失
    此函数会输出符合要求的最小的kernel_size和stride
    :param config: 期望的每个block的channel
    :param image_size: 输入的图片大小
    :param output_size: 期望的输出的特征大小
    :param min_kernel_size: 期望的最小的kernel_size
    :param max_kernel_size: 期望的最大的kernel_size
    :param min_stride: 期望的最小的stride
    :param max_stride: 期望的最大的stride
    :return: 符合要求的一个layer配置解
    '''

    mmap_size = image_size
    layer_number = (len(block_config)-1)*2
    mmap = np.zeros((layer_number+1, mmap_size+1)).astype(int)
    mmap[0][image_size] = 1
    pre = np.zeros((layer_number + 1, mmap_size + 1)).astype(int)
    layers = [[] for i in range(layer_number+1)]
    for i in range(len(layers)):
        for j in range(mmap_size+1):
            layers[i].append(Layer())

    for i in range(layer_number):
        for j in range(mmap_size+1):
            if mmap[i][j] == 1:
                for f in range(max_kernel_size, min_kernel_size-1, -1):
                    for s in range(max_stride, min_stride-1, -1):
                        # 保证不丢失特征
                        if (j-f) % s == 0:
                            t_output_size = calc_output_size(j, 0, f, s)
                            if t_output_size >= output_size and t_output_size <= image_size:
                                mmap[i+1][t_output_size] = 1
                                layers[i+1][t_output_size] = Layer(f, s)
                                pre[i+1][t_output_size] = j

    ans_layers = []
    if mmap[layer_number][output_size] == 1:
        pre_no = output_size
        for i in range(layer_number, 0, -1):
            ans_layers.append(layers[i][pre_no])
            pre_no = pre[i][pre_no]

    ans_layers.reverse()

    return ans_layers


def show_mid_feature_size(image_size, layers):
    feature_size = image_size
    for i in range(len(layers)):
        f, s = layers[i].kernel_size, layers[i].stride
        feature_size = calc_output_size(feature_size, 0, f, s)
        print(feature_size)


if __name__ == '__main__':
    config = [1, 5, 10]
    image_size = 28
    output_size = 1
    # layers = imperfect_build_moudel(config, image_size, output_size)
    layers = perfect_build_moudel(config, image_size, output_size)
    for i in range(len(layers)):
        print(layers[i])

    show_mid_feature_size(image_size, layers)