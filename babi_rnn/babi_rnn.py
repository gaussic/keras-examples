'''Trains two recurrent neural networks based upon a story and a question.
The resulting merged vector is then queried to answer a range of bAbI tasks.
The results are comparable to those for an LSTM model provided in Weston et al.:
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
http://arxiv.org/abs/1502.05698

基于一个故事和一个问题，训练两个循环神经网络。
它们的结果合并的向量在之后被用来回答一系列的 bAbI 任务。
结果与 Weston在这篇论文中提到的一个 LSTM 模型具有一定的可比性：
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
http://arxiv.org/abs/1502.05698

任务类型                      | FB LSTM 基准线   | Keras QA
Task Number                  | FB LSTM Baseline | Keras QA
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               | 100.0
QA2 - Two Supporting Facts   | 20               | 50.0
QA3 - Three Supporting Facts | 20               | 20.5
QA4 - Two Arg. Relations     | 61               | 62.9
QA5 - Three Arg. Relations   | 70               | 61.9
QA6 - Yes/No Questions       | 48               | 50.7
QA7 - Counting               | 49               | 78.9
QA8 - Lists/Sets             | 45               | 77.2
QA9 - Simple Negation        | 64               | 64.0
QA10 - Indefinite Knowledge  | 44               | 47.7
QA11 - Basic Coreference     | 72               | 74.9
QA12 - Conjunction           | 74               | 76.4
QA13 - Compound Coreference  | 94               | 94.4
QA14 - Time Reasoning        | 27               | 34.8
QA15 - Basic Deduction       | 21               | 32.4
QA16 - Basic Induction       | 23               | 50.6
QA17 - Positional Reasoning  | 51               | 49.1
QA18 - Size Reasoning        | 52               | 90.8
QA19 - Path Finding          | 8                | 9.0
QA20 - Agent's Motivations   | 91               | 90.7
For the resources related to the bAbI project, refer to:
https://research.facebook.com/researchers/1543934539189348
想了解更多关于 bAbI project 的资源，访问：
https://research.facebook.com/researchers/1543934539189348

Notes:
注意：

- With default word, sentence, and query vector sizes, the GRU model achieves:
  - 100% test accuracy on QA1 in 20 epochs (2 seconds per epoch on CPU)
  - 50% test accuracy on QA2 in 20 epochs (16 seconds per epoch on CPU)
In comparison, the Facebook paper achieves 50% and 20% for the LSTM baseline.
- 使用默认的词，句子和查询向量尺寸，GRU 模型 获得了：
  - 100% 测试准确率，在 QA1 上进行 20 轮迭代（在 CPU 上每轮迭代 2 秒）
  - 50% 测试准确率，在 QA2 上进行 20 轮迭代（在 CPU 上每轮迭代 16 秒）
相比之下，Facebook 的论文使用 LSTM 基准线分别获得了 50% 和 20% 的准确率。

- The task does not traditionally parse the question separately. This likely
improves accuracy and is a good example of merging two RNNs.
- 这一任务没有传统地单独地解析问题。这似乎提升了准确率，并且是一个很好的合并两个 RNN 的例子。

- The word vector embeddings are not shared between the story and question RNNs.
- 词向量嵌入在故事和问题 RNN 之间不是共享的。

- See how the accuracy changes given 10,000 training samples (en-10k) instead
of only 1000. 1000 was used in order to be comparable to the original paper.
- 可以看看1000个训练样本时（en-10k）的准确率情况，而不仅仅是1000个。 1000的目的是用来和论文进行对比。

- Experiment with GRU, LSTM, and JZS1-3 as they give subtly different results.
- 用 GRU、LSTM 和 JZS1-3 分别试验，因为它们给出了微妙的不同的结果。

- The length and noise (i.e. 'useless' story components) impact the ability for
LSTMs / GRUs to provide the correct answer. Given only the supporting facts,
these RNNs can achieve 100% accuracy on many tasks. Memory networks and neural
networks that use attentional processes can efficiently search through this
noise to find the relevant statements, improving performance substantially.
This becomes especially obvious on QA2 and QA3, both far longer than QA1.
长度和噪声（即，“无用”的故事组件）影响了 LSTMs /GRUs 提供正确答案的能力。如果只给出支持的事实，
这些 RNNs 可以在多项任务中获得 100% 的准确率。记忆网络和使用注意力机制的神经网络可以高效地从这
些噪声中搜索到相关的陈诉，大大提高性能。这在 QA2 和 QA3 上变得非常明显，远大于 QA1。
'''

from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    返回包含标点符号的句子标记
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format.

    If only_supporting is true, only the sentences that support the answer are kept.

    解析bAbi任务格式中提供的故事,
    如果 only_supporting 为 true，那么只有支持答案的句子被保留。
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                # 只选取与问题答案相关的子故事
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                # 提供全部子故事
                substory = [x for x in story if x]
            data.append((substory, q, a))   # 数据的组织形式是 (子故事, 问题，答案）
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.

    给出文件名，读文件，检索故事，然后将句子转化为单个的故事。如果提供了最大长度，那么任何长于 max_length 的故事都被丢弃。
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)  # 将列表展平

    # 将 data 中的 story 展平
    data = [(flatten(story), q, answer) for story, q,
            answer in data if not max_length or len(flatten(story)) < max_length]
    return data

# 向量化操作
def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved 不要忘记索引 0 是被保留的，所以 y 会多一位
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    # 填充
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

# 模型参数
RNN = recurrent.LSTM      # 使用 LSTM 模型
EMBED_HIDDEN_SIZE = 50    # 潜入层大小
SENT_HIDDEN_SIZE = 100    # 句子隐藏层大小
QUERY_HIDDEN_SIZE = 100   # 查询隐藏层大小
BATCH_SIZE = 32           # 批数据量
EPOCHS = 40               # 迭代次数
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))

try:
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)
# Default QA1 with 1000 samples  默认QA1 1000个样本
challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples  QA1 10000个样本
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples  QA2 1000个样本
# challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# QA2 with 10,000 samples   QA2 10000个样本
# challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
train = get_stories(tar.extractfile(challenge.format('train')))  # 训练集
test = get_stories(tar.extractfile(challenge.format('test')))    # 测试集

# 词汇表
vocab = sorted(reduce(lambda x, y: x | y, (set(
    story + q + [answer]) for story, q, answer in train + test)))
# Reserve 0 for masking via pad_sequences # 保留 0 位作为填充词
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))  # 词汇与 id 的映射
story_maxlen = max(map(len, (x for x, _, _ in train + test)))    # 最长故事长度
query_maxlen = max(map(len, (x for _, x, _ in train + test)))    # 最长查询长度

X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)    # 训练样本
tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)  # 测试样本

print('vocab = {}'.format(vocab))
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')
# 构建句子 RNN 模型
sentrnn = Sequential()
sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                      input_length=story_maxlen))
sentrnn.add(Dropout(0.3))

# 构建查询 RNN 模型
qrnn = Sequential()
qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                   input_length=query_maxlen))
qrnn.add(Dropout(0.3))
qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
qrnn.add(RepeatVector(story_maxlen))

# 合并以上两个模型
model = Sequential()
model.add(Merge([sentrnn, qrnn], mode='sum'))  # 合并操作，将两个模型的输出相加
model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(vocab_size, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([X, Xq], Y, batch_size=BATCH_SIZE,
          nb_epoch=EPOCHS, validation_split=0.05)
loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
