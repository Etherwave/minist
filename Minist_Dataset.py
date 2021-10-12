import struct
import numpy as np
import cv2
from torch.utils.data import Dataset

class MinistDataset(Dataset):

    def __init__(self, minist_dataset_path="/home/amzing/dataset/", mode='train'):
        self.mode = mode
        self.minist_dataset_path = minist_dataset_path

        # 训练集文件
        self.train_images_idx3_ubyte_file = self.minist_dataset_path + 'MNIST/raw/train-images-idx3-ubyte'
        # 训练集标签文件
        self.train_labels_idx1_ubyte_file = self.minist_dataset_path + 'MNIST/raw/train-labels-idx1-ubyte'

        # 测试集文件
        self.test_images_idx3_ubyte_file = self.minist_dataset_path + 'MNIST/raw/t10k-images-idx3-ubyte'
        # 测试集标签文件
        self.test_labels_idx1_ubyte_file = self.minist_dataset_path + 'MNIST/raw/t10k-labels-idx1-ubyte'

        self.train_images = self.decode_idx3_ubyte(self.train_images_idx3_ubyte_file)
        self.train_labels = self.decode_idx1_ubyte(self.train_labels_idx1_ubyte_file)

        self.test_images = self.decode_idx3_ubyte(self.test_images_idx3_ubyte_file)
        self.test_labels = self.decode_idx1_ubyte(self.test_labels_idx1_ubyte_file)

        self.size = 0
        if self.mode == "train":
            self.size = len(self.train_images)
        elif self.mode == "test":
            self.size = len(self.test_images)



    def decode_idx1_ubyte(self, idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx1_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            # if (i + 1) % 10000 == 0:
            #     print('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels

    def decode_idx3_ubyte(self, idx3_ubyte_file):
        """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx3_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
        fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
        images = np.empty((num_images, num_rows, num_cols))
        for i in range(num_images):
            # if (i + 1) % 10000 == 0:
            #     print('已解析 %d' % (i + 1) + '张')
            #     print(offset)
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            offset += struct.calcsize(fmt_image)
        return images

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image, label = None, None
        if self.mode == "train":
            image, label = self.train_images[index], self.train_labels[index]
        elif self.mode == "test":
            image, label = self.test_images[index], self.test_labels[index]
        if isinstance(image, np.ndarray):
            image = np.expand_dims(image, axis=0)
        return image, label

if __name__ == '__main__':
    mdata = MinistDataset()
    for i in range(100):
        print(mdata.train_labels[i])
        print(np.shape(mdata.train_images[i]))
        cv2.imshow("test", mdata.train_images[i])
        cv2.imwrite(f"{i}.jpg", mdata.train_images[i])
        cv2.waitKey()