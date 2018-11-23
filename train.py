import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # 结巴分词
# gensim用来加载预训练词向量
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
#定义4类“情感倾向值”标签
SentimentalLabels=['pos','neu','neg','not']
#使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)
# 由此可见每一个词都对应一个长度为300的向量
embedding_dim = cn_model['浙江大学'].shape[0]
# 打开文件
workbook = xlrd.open_workbook(r'trainSet.xlsx.xlsx')#赛题中选用sentiment_analysis_trainingset.csv
worksheet1 = workbook.sheet_by_name(u'trainset')
#print(worksheet1.cell_value(1,1))
table = [([0] * 22) for i in range(15000)]
trainTextsOrig = [] # 存储所有评价，每例评价为一条string
#读入一整张表
for row in range(0,15000):
    for col in range(0,22):
        table[row][col]=worksheet1.cell_value(row+1,col)
        #print(table[row][col])
print(table)
def sTrain(sFactor,sLabel,trainTextsOrig,flag):
    # 进行分词和tokenize
    # train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
    train_tokens = []
    for text in trainTextsOrig:
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
    plt.hist(np.log(num_tokens), bins = 100)
    plt.xlim((0,10))
    plt.ylabel('number of tokens')
    plt.xlabel('length of tokens')
    plt.title('Distribution of tokens length')
    plt.show()
    # 取tokens平均值并加上两个tokens的标准差，
    # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    max_tokens
    # 取tokens的长度时，大约95%的样本被涵盖
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
    trainTextsOrig[0]
    # 只使用前50000个词
    num_words = 50000
    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
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
    # 准备target向量，
    train_target = np.concatenate( (np.ones(flag),np.zeros(14999-flag)) )
    #############################
    # 进行训练和测试样本的分割
    from sklearn.model_selection import train_test_split
    # 90%的样本作为训练集，剩余10%作为测试集
    X_train, X_test, y_train, y_test = train_test_split(train_pad,train_target,test_size=0.1,random_state=12)
    # 查看训练样本，确认无误
    print(reverse_tokens(X_train[35]))
    print('class: ',y_train[35])
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
    checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',verbose=1, save_weights_only=True,save_best_only=True)
    # 尝试加载已训练模型
    try:
        model.load_weights(path_checkpoint)
    except Exception as e:
        print(e)
    # 定义early stoping如果3个epoch内validation loss没有改善则停止训练
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    # 自动降低learning rate
    lr_reduction = ReduceLROnPlateau(monitor='val_loss',factor=0.1, min_lr=1e-5, patience=0,verbose=1)
    # 定义callback函数
    callbacks = [
        earlystopping, 
        checkpoint,
        lr_reduction
    ]
    # 开始训练
    model.fit(X_train, y_train,validation_split=0.1, epochs=1,batch_size=128,callbacks=callbacks)
    result = model.evaluate(X_test, y_test)
    print('Accuracy:{0:.2%}'.format(result[1]))

for sFactor in range(1,21):
    for sLabel in range(0,3):
        # 添加完所有样本之后，trainTextsOrig为一个含有15000条文本的list
        # flag记录识别的评论量
        flag=0
        trainTextsOrig=[]
        for i in range(1,15000):
            #print(table[i][5])
            if table[i][sFactor+1]==(1-sLabel):
                trainTextsOrig.append(table[i][1])
                flag=flag+1
        for i in range(1,15000):
            if table[i][sFactor+1]!=(1-sLabel):
                trainTextsOrig.append(table[i][1])
        len(trainTextsOrig)
        sTrain(sFactor,sLabel,trainTextsOrig,flag)
