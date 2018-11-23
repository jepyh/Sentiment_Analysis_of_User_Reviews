import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
# 我们使用tensorflow的keras接口来建模
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import xlrd
import xlwt
import xlutils
from xlutils.copy import copy
#定义4类“情感倾向值”标签
SentimentalLabels=['pos','neu','neg','not']
#使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)
# 由此可见每一个词都对应一个长度为300的向量
embedding_dim = cn_model['浙江大学'].shape[0]
import os
# 打开文件
workbook = xlrd.open_workbook(r'trainSet.xlsx')#赛题中选用sentiment_analysis_trainingset.csv
worksheet1 = workbook.sheet_by_name(u'trainset')
#print(worksheet1.cell_value(1,1))
#获得的计算分割权重
wCoef=[[0.16423324,0.20563854,0.1832024,0.05415282,0.34744427,0.04391111,0.08067036,0.15416728,0.14781696,0.19098693,0.5410546,0.2455689,0.22419062,0.26717985,0.29366508,0.5631437,0.22864255,0.196232,0.5745875,0.47224596],
[0.012059669,0.007132139,0.025371287,0.0393093,0.13689324,0.017043034,0.028650275,0.3116929,0.027504813,0.20021997,0.101439446,0.0496168,0.08692318,0.046705656,0.09374869,0.3830413,0.053464673,0.022211837,0.40328702,0.049615122],
[0.014511555,0.008624982,0.04348241,0.039011925,0.123212956,0.014857611,0.07430692,0.14189564,0.03250708,0.020250684,0.01927395,0.031986337,0.06859584,0.052187722,0.11240926,0.06891142,0.036198065,0.024704598,0.1760263,0.117662795]]
table = [([0] * 22) for i in range(15000)]
train_texts_orig = [] # 存储所有评价，每例评价为一条string
#读入一整张表
for row in range(0,15000):
    for col in range(0,22):
        table[row][col]=worksheet1.cell_value(row+1,col)
        #print(table[row][col])

def sPredict(sFactor,sLabel,test_list,train_texts_orig,flag):
    # 进行分词和tokenize
    # train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
    train_tokens = []
    for text in train_texts_orig:
        # 去掉标点
        text = re.sub("[/s+/./!//_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)#[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]
        # 结巴分词
        cut = jieba.cut(text)
        # 结巴分词的输出结果为一个生成器
        # 把生成器转换为list
        cut_list = [ i for i in cut ]
        for i, word in enumerate(cut_list):
            try:
                # 将词转换为索引index
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                # 如果词不在字典中，则输出0
                cut_list[i] = 0
        train_tokens.append(cut_list)
    # 获得所有tokens的长度
    num_tokens = [ len(tokens) for tokens in train_tokens ]
    num_tokens = np.array(num_tokens)
    # 平均tokens的长度
    np.mean(num_tokens)
    # 最长的评价tokens的长度
    np.max(num_tokens)
    # 取tokens平均值并加上两个tokens的标准差，
    # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    max_tokens
    # 取tokens的长度为236时，大约95%的样本被涵盖
    # 我们对长度不足的进行padding，超长的进行修剪
    np.sum( num_tokens < max_tokens ) / len(num_tokens)
    # 用来将tokens转换为文本
    def reverse_tokens(tokens):
        text = ''
        for i in tokens:
            if i != 0:
                text = text + cn_model.index2word[i]
            else:
                text = text + ' '
        return text
    reverse = reverse_tokens(train_tokens[0])
    # 原始文本
    train_texts_orig[0]
    embedding_dim
    # 只使用前50000个词
    num_words = 50000
    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 50000 * 300
    for i in range(num_words):
        embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    # 检查index是否对应，
    # 输出300意义为长度为300的embedding向量一一对应
    np.sum( cn_model[cn_model.index2word[333]] == embedding_matrix[333] )
    # embedding_matrix的维度，
    # 这个维度为keras的要求，后续会在模型中用到
    embedding_matrix.shape
    # 进行padding和truncating， 输入的train_tokens是一个list
    # 返回的train_pad是一个numpy array
    train_pad = pad_sequences(train_tokens, maxlen=max_tokens,padding='pre', truncating='pre')
    # 超出五万个词向量的词用0代替
    train_pad[ train_pad>=num_words ] = 0
    # 可见padding之后前面的tokens全变成0，文本在最后面
    train_pad[33]
    ############################
    # 用LSTM对样本进行分类
    model = Sequential()
    # 模型第一层为embedding
    model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_tokens,trainable=False))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    # 我们使用adam以0.001的learning rate进行优化
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    # 我们来看一下模型的结构，一共90k左右可训练的变量
    model.summary()
    # 建立一个权重的存储点
    path_checkpoint = 'sentiment_checkpoint_'+str(SentimentalLabels[sLabel])+str(sFactor)+'.keras'
    print("Now,the model is")
    print(str(SentimentalLabels[sLabel]))
    print(str(sFactor))
    # 尝试加载已训练模型
    try:
        model.load_weights(path_checkpoint)
    except Exception as e:
        print(e)
    print('wCoef:')    
    print(wCoef[sLabel][sFactor-1])
    row=0
    for text in test_list:
        row = row+1
        print(row)
        # 去标点
        text = re.sub("[/s+/./!//_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        # 分词
        cut = jieba.cut(text)
        cut_list = [ i for i in cut ]
        # tokenize
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
                if cut_list[i]>50000:
                    cut_list[i]=0
            except KeyError:
                cut_list[i] = 0
        # padding
        tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,padding='pre',truncating='pre')
        #print(tokens_pad)
        # 预测
        result = model.predict(x=tokens_pad)
        coef = result[0][0]
        #计算预测结果
        if coef >= wCoef[sLabel][sFactor-1]:
            ws.write(row, (sFactor+1), (1-sLabel))
            print(1-sLabel)     
# 打开预测集
workbook2 = xlrd.open_workbook(r'testSet.xls')#赛题中选用sentiment_analysis_testb.csv
worksheet2 = workbook2.sheet_by_name(u'Sheet1')
# 将操作文件对象拷贝，变成可写的workbook对象
new_excel = copy(workbook2)
# ws获得最终结果
ws = new_excel.get_sheet(0)
#print(ws)
dataSize = 2000 #200000
test_list = [([0]) for i in range(dataSize)]
#读入一整张表
for row in range(0,dataSize):
    test_list[row] = worksheet2.cell_value(row+1,1)
for sFactor in range(1,21):
    for sLabel in range(0,3):
        flag=0
        train_texts_orig=[]
        for i in range(1,15000):
            #print(table[i][5])
            if table[i][sFactor+1]==(1-sLabel):
                train_texts_orig.append(table[i][1])
                flag=flag+1
        for i in range(1,15000):
            if table[i][sFactor+1]!=(1-sLabel):
                train_texts_orig.append(table[i][1])
        len(train_texts_orig)
        print("sFactor is")
        print(sFactor)
        print("sLabel is")
        print(sLabel)
        sPredict(sFactor,sLabel,test_list,train_texts_orig,flag)
# 另存为最终结果
new_excel.save('finalResults.xls')
