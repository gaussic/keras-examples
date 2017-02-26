
#%%
'''
An implementation of sequence to sequence learning for performing addition
使用Sequence To Sequence Learning 来做加法的一个实现

Input: "535+61"
Output: "596"
输入："535+61"
输出："596"

Padding is handled by using a repeated sentinel character (space)
填充操作使用重复的哨兵字符（空格）实现

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"：http://arxiv.org/abs/1410.4615 and 
"Sequence to Sequence Learning with Neural Networks": 
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
输入可以选择性地反转，被认为可以提升很多任务的性能，例如：
"Learning to Execute"：http://arxiv.org/abs/1410.4615 和
"Sequence to Sequence Learning with Neural Networks": 
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
理论上，它引入和源和目标之前的短期依赖性。

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs
反转两位数字：
+ 一个LSTM层（128个隐藏层神经元），在55轮迭代后，5K的训练样本 = 99% 训练/测试准确率

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs
反转三位数字：
+ 一个LSTM层（128个隐藏层神经元），在100轮迭代后，50K的训练样本 = 99% 训练/测试准确率

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs
反转四位数字：
+ 一个LSTM层（128个隐藏层神经元），在20轮迭代后，400K的训练样本 = 99% 训练/测试准确率

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
反转五位数字：
+ 一个LSTM层（128个隐藏层神经元），在30轮迭代后，550K的训练样本 = 99% 训练/测试准确率
'''

from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output

    给定一个字符集合：
    将它们编码为 one hot 表示
    将 one hot 表示解码为字符输出
    将概率向量（类别）解码为字符输出
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.

        初始化字符表。
        # 参数：
            chars: 可出现在输入中的字符。
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.

        字符串C的 one hot 编码。

        # 参数：
            num_rows：返回的 one hot 编码的行数，用于保持每个数据的行数相同。
        # 返回：
            X：[num_rows, len(self.chars)]，总共 num_rows 行，每一行表示一个字符的 one hot 编码
        """
        X = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        """将 one hot 编码或者概率值(类别)向量解码为字符输出

        # 参数：
            calc_argmax: 是否为 one-hot
        """
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset.
# 模型和数据集的参数
TRAINING_SIZE = 50000  # 训练样本大小
DIGITS = 3             # 数字位数
INVERT = True          # 是否可反转

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of int is DIGITS.
# 输入的最大长度是 'int + int' (即, '345+678'). int 的最大长度是 DIGITS。
MAXLEN = DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding.
# 所有的数字，加号，以及用于填充的空格。
chars = '0123456789+ '
ctable = CharacterTable(chars)

# 生成训练数据
questions = []  # 问题
expected = []  # 正确答案
seen = set()   # 是否出现过
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                            for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that X+Y == Y+X (hence the sorting).
    # 跳过所有已经出现的问题
    # 同样跳过任何 X + Y == Y + X 的问题 (因此要排序)
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    # 填充数据，使得长度均为 MAXLEN。
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1.
    # 答案的最大长度为 DIGITS + 1。
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the space used for padding.)
        # 反转问题，即，'12+345  ' 变成 '  543+21'。(注意填充用的空格)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

# 向量化操作，将训练样本转化为 one hot 表示
# X：[(len(questions), MAXLEN, len(chars)]，每个样本的问题都是固定长度为 MAXLEN 的序列
# y：[(len(questions), DIGITS + 1, len(chars)]，每个样本的答案都是固定长度为 DIGITS + 1 的序列
print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits.
# 将 (X, y) 打乱，因为 X 的后面部分几乎都是更大的整数
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
# 明确设定 10% 的验证数据，并且从不训练它们。
split_at = len(X) - len(X) // 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print('Training Data:')
print(X_train.shape)
print(y_train.shape)

print('Validation Data:')
print(X_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
# 这里是模型的参数，试着将 RNN 换成 GRU 或者 SimpleRNN。
RNN = recurrent.LSTM   # 使用 LSTM
HIDDEN_SIZE = 128      # 隐藏层神经元
BATCH_SIZE = 128       # 批处理尺寸
LAYERS = 1             # 层数

# 构建模型
print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
# 使用 RNN 将输入序列编码，产生一个 HIDDEN_SIZE 大小的输出
# 注意：在输入序列长度可变的情况下，使用 input_shape=(None, nb_feature)。
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
# 在每一个时间步，重复提供RNN的最后隐藏状态，作为解码器 RNN 的输入。
# 重复 'DIGITS + 1' 次，因为这是输出的最大长度，即，当 DIGITS=3 时，最大输出是：999+999=1998。
model.add(RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer.
# 编码器 RNN 可以是多层堆叠，或者仅仅一层。
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (nb_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    # 通过设定 return_sequences 为 True，返回的不仅仅是最后一次的输出，而是目前为止的整个输出，
    # 其形式为 (nb_samples, timesteps, output_dim)。
    # 这很必要，因为下面的 TimeDistributed 需要第一个维度为 timesteps。
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
# 对输入的每一个时间片应用一个稠密层(全连接层)。对于输出序列的每一步，决定应该选择哪一个字符。
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()  # 输出模型的概述

# Train the model each generation and show predictions against the validation
# dataset.
# 在每一轮迭代中，训练模型并且输出验证集的预测结果
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    # 随机从验证集中选择 10 个样本，用以可视化误差。
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)  # 输出预测分类（序列）
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        if correct == guess:
            print(colors.ok + 'v' + colors.close, end=" ")
        else:
            print(colors.fail + 'x' + colors.close, end=" ")
        print(guess)
        print('---')
