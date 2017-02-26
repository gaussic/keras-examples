'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

在MNIST数据集上训练一个简单的深度神经网络。
在20轮迭代后获得了 98.40% 的测试准确率
(还有很多参数调优的余地)。
在 K520 GPU上，每轮迭代 2 秒。
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility 为了重现结果

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


# 模型参数
batch_size = 128  # 批大小
nb_classes = 10   # 类别
nb_epoch = 20     # 迭代轮次

# the data, shuffled and split between train and test sets
# 经过打乱和切分的训练与测试数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 将 28x28 的图像展平成一个 784 维的向量
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 规范化
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 将分类向量转化为二元分类矩阵(one hot)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 模型定义
model = Sequential()  # 序列化模型
model.add(Dense(512, input_shape=(784,)))  # 全连接层，784维输入，512维输出
model.add(Activation('relu'))              # relu 激活函数
model.add(Dropout(0.2))                    # Dropout 舍弃部分因曾结点的权重，防止过拟合
model.add(Dense(512))                      # 由一个全连接层，512维输出
model.add(Activation('relu'))              # relu 激活函数
model.add(Dropout(0.2))                    # Dropout 舍弃部分因曾结点的权重，防止过拟合
model.add(Dense(10))                       # 由一个全连接层，10维输出
model.add(Activation('softmax'))           # softmax 激活函数用于分类

model.summary()    # 模型概述

# 定义模型的损失函数，优化器，评估矩阵
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 训练，迭代 nb_epoch 次，使用验证集验证
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

# 评估测试集测试误差，准确率
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])