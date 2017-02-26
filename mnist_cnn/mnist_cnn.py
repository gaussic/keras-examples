'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

在 MNIST 数据集上训练一个简单的卷积神经网络。
在 12 轮迭代后，的到了99.25% 的测试准确率。
(还有很多参数调优的余地)。
在 K520 网格 GPU上，每轮迭代需要 16 秒。
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility 为了重现结果

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# 模型参数
batch_size = 128  # 批大小
nb_classes = 10   # 类别
nb_epoch = 12     # 迭代轮次

# input image dimensions
# 输入图像的维度
img_rows, img_cols = 28, 28
# number of convolutional filters to use
# 要使用的卷积核的个数
nb_filters = 32
# size of pooling area for max pooling
# 最大池化的面积尺寸
pool_size = (2, 2)
# convolution kernel size
# 卷积核尺寸
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
# 经过打乱和切分的训练与测试数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Theano 和 Tensorflow 的输入维度和顺序不一样，需要使用转换为合适的维度顺序
# Theano 的维度：[1, 28, 28]
# Tensorflow 的维度: [28, 28, 1]
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 标准化
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 将分类向量转化为二元分类矩阵(one hot)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 模型定义
model = Sequential()
# 2D 卷积神经网络，使用 nb_filters 个卷积核
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))    # relu激活函数
# 2D 卷积神经网络，使用 nb_filters 个卷积核
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))    # relu激活函数
model.add(MaxPooling2D(pool_size=pool_size))  # 池化层，使用最大池化
model.add(Dropout(0.25))        # Dropout 防止过拟合

model.add(Flatten())             # 压平，将多维输入转变为一维
model.add(Dense(128))            # 全连接层，128维
model.add(Activation('relu'))    # relu激活函数
model.add(Dropout(0.5))          # Dropout 防止过拟合
model.add(Dense(nb_classes))     # 全连接层，维度为分类的个数
model.add(Activation('softmax')) # softmax 激活函数，用于分类

model.summary()  # 模型概述

# 定义模型的损失函数，优化器，评估矩阵
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# 训练，迭代 nb_epoch 次，使用验证集验证
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

 # 评估测试集测试误差，准确率
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
