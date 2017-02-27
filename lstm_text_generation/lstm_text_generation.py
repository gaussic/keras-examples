'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.


使用LSTM语言模型生成文本

这是根据尼采的作品生成文本的样例，语料是字符级的。
在生成的文本听起来连贯之前，至少需要 20 轮迭代。
推荐使用 GPU 来运行这个脚本，因为 循环神经网络的计算是非常密集的。
如果你想在新的数据上测试这个脚本，请确保你的语料至少了 ~100k 的字符数量，~1M 更好。
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

# 读取尼采作品的文本数据
path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

# 生成字符词汇表，字符与索引之间的映射
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# 将文本切割为半冗余的序列，长度为 maxlen 个字符
maxlen = 40        # 序列长度
step = 3           # 步长
sentences = []     # 句子
next_chars = []    # 句子的下一个字符
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

# 向量化操作
# X: [nb_sequences, maxlen， len(chars)]，字符的 one-hot 表示
# y：[nb_sequences, len(chars)]，字符的 one-hot 表示
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 由于在这个脚本中，是以字符级别的 LSTM 来生成文本，
# 字符词汇表非常小，所以并没有使用词嵌入
# 但是处理其他的文本，例如中文，或者单词级别的 LSTM 的话，
# 词汇表相当大，这个时候使用词嵌入进行优化是非常合适的。

# build the model: a single LSTM
# 构建模型，使用单层 LSTM 循环神经网络
print('Build model...')
model = Sequential()
# 输入为一个序列，输出为 128维的向量，即 LSTM 的最终状态
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))      # 全连接层，长度为词汇表大小
model.add(Activation('softmax'))  # softmax激活层，检查下一个最有可能的字符 

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 根据预测输入抽样
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # 从预测结果进一步抽样下一个字符，这里 temperature 决定了输出的多样性
    # 具体可以参照以下两篇中的解释
    # http://home.deib.polimi.it/restelli/MyWebSite/pdf/rl5.pdf
    # http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node17.html
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
# 训练模型，在每一次迭代后输出生成的文本
for iteration in range(1, 60): # 59轮迭代
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)  # 训练

    # 随机选择起始位置，以这个起始位置的文本为基础生成后续文本
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        # 根据选择的起始位置检出起始文本，这里称为种子文本
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)   # 打印文本

        for i in range(400):   # 连续生成 400 个后续字符
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]  # 预测下一个结果
            next_index = sample(preds, diversity)   # 抽样出下一个字符的索引值
            next_char = indices_char[next_index]    # 检出下一个字符

            generated += next_char
            sentence = sentence[1:] + next_char   # 输入后移一格

            sys.stdout.write(next_char)  # 连续打印
            sys.stdout.flush()           # 刷新控制台
        print()