'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

这个例子演示了使用 Convolution1D 进行文本情感分类（这是一个二元分类，结果只有0和1）。
在 2 轮迭代后，得到了 0.89 的测试准确率。
在 Intel i5 2.4Ghz CPU 上的速度是 90s/epoch。
在 Tesla K40 GPU 上的速度是 10s/epoch。

（这个例子没有划分验证集，直接使用测试集做验证，并不是非常的正规。）
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb


# set parameters:  设定参数
max_features = 5000  # 最大特征数（词汇表大小）
maxlen = 400         # 序列最大长度
batch_size = 32      # 每批数据量大小
embedding_dims = 50  # 词嵌入维度
nb_filter = 250      # 1维卷积核个数
filter_length = 3    # 卷积核长度
hidden_dims = 250    # 隐藏层维度
nb_epoch = 10        # 迭代次数

# 载入 imdb 数据
print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# 样本填充到固定长度 maxlen，在每个样本前补 0 
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# 构建模型
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# 先从一个高效的嵌入层开始，它将词汇的索引值映射为 embedding_dims 维度的词向量
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
# 添加一个 1D 卷积层，它将学习 nb_filter 个 filter_length 大小的词组卷积核
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
# 使用最大池化
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
# 添加一个原始隐藏层
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
# 投影到一个单神经元的输出层，并且使用 sigmoid 压缩它
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()  # 模型概述

# 定义损失函数，优化器，评估矩阵
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练，迭代 nb_epoch 次
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
