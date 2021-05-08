'''
Date: 2021-04-13 14:40:13
Author: ZHR
Description: 使用朴素贝叶斯过滤垃圾邮件；垃圾邮件：spam；非垃圾邮件文件夹：ham，各有25封邮件
LastEditTime: 2021-04-19 15:29:00
'''
# -*- coding: utf-8 -*-
import re
import numpy as np
import pickle


# 准备数据：切分文本
def textParse(bigString):
    # listOfTokens = re.split(r'\W*', bigString)  # 匹配非字母数字下划线
    listOfTokens = re.split(r'\W', bigString)  # 匹配非字母数字下划线
    return [tok.lower() for tok in listOfTokens
            if len(tok) > 2]  # 若文本中有URL，对其进行切分时，会得到很多词，为避免该情况，限定字符创的长度


# 将文档矩阵中的所有词构成词汇表
def creatVocabList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)  # 两个集合的并集
    return list(vocabSet)


# 将某一文档转换成词向量，该向量中所含数值数目与词汇表中词汇数目相同
# 词袋模型
def bagOfWords2Vec(vocabList, inputSet):  # 参数分别为词汇表，输入文档
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 朴素贝叶斯分类器训练函数
# trainMatrix为文档词向量矩阵,
# trainCategory为每篇文档的类标签构成的向量
# p(ci|w)=p(w|ci)*p(ci)/p(w) 假设使其判定为垃圾邮件的每个标签Wi互相独立
# 则有p(w|ci)=p(w0|ci)p(w1|ci)...p(wN|ci)
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 总文档数
    numWords = len(trainMatrix[0])  # 所有词的数目
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱性概率，即P(Ci)
    p0Num = np.ones(numWords)  # 避免某一个特征为0导致总结果为0初始化为1
    p1Num = np.ones(numWords)
    p0Deom = 2.0
    p1Deom = 2.0
    '''
    避免某一个特征为0导致总结果为0,故采用拉普拉斯变换平滑，将分子pnum初始化为1
    将分母pdeom初始化为标签个数2
    '''
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 向量相加
            p1Deom += sum(trainMatrix[i])  # 所有垃圾邮件中出现的词条的总计数值
        else:
            p0Num += trainMatrix[i]
            p0Deom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Deom)  # 在垃圾文档条件下词汇表中单词的出现概率
    p0Vect = np.log(p0Num / p0Deom)  # 采用对数是为了解决下溢问题
    # pAbusive就是文档属于垃圾文档的概率p(Ci)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
# # 参数分别为：要分类的向量以及使用trainNB0()计算得到的三个概率
# # 因为将概率转化为log对数形式了，所以p(w|ci)=p(w0|ci)p(w1|ci)...p(wN|ci)应该转为对数的求和
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass)
    if p1 > p0:
        return 1  # 表示侮辱性文档
    else:
        return 0


# 测试算法：使用朴素贝叶斯交叉验证。同时保存分类模型的词汇表以及三个概率值，避免判断时重复求值
def spamTest():
    test_error = []
    docList = []  # 文档（邮件）矩阵
    classList = []  # 类标签列表
    for i in range(1, 26):
        wordlist = textParse(open('task1/spam/{}.txt'.format(str(i))).read())
        docList.append(wordlist)
        classList.append(1)
        wordlist = textParse(open('task1/ham/{}.txt'.format(str(i))).read())
        docList.append(wordlist)
        classList.append(0)
    vocabList = creatVocabList(docList)  # 所有邮件内容的词汇表
    file = open('task1/result/vocabList.txt', mode='wb')  # 存储词汇表
    pickle.dump(vocabList, file)
    file.close()
    # 对需要测试的邮件，根据其词表fileWordList构造向量
    # 随机构建40训练集与10测试集
    trainingSet = list(range(50))  # 训练集的索引列表
    testSet = []  # 测试集的索引列表
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []  # 训练集
    trainClasses = []  # 训练集中向量的类标签列表
    for docIndex in trainingSet:
        # 使用词袋模式构造的向量组成训练集
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pAb = trainNB0(trainMat, trainClasses)
    file = open('task1/result/threeRate.txt', mode='wb')  # 用以存储分类器的三个概率
    pickle.dump([p0v, p1v, pAb], file)
    file.close()
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0v, p1v, pAb) != classList[docIndex]:
            errorCount += 1
            if docIndex in test_error:
                pass
            else:
                test_error.append(docIndex)
    return (float(errorCount) / len(testSet)), test_error


# 构造分类器
def fileClassify(filepath):
    fileWordList = textParse(open(filepath, mode='r').read())
    file = open('task1/result/vocabList.txt', mode='rb')
    vocabList = pickle.load(file)
    vocabList = vocabList
    fileWordVec = bagOfWords2Vec(vocabList, fileWordList)  # 被判断文档的向量
    file = open('task1/result/threeRate.txt', mode='rb')
    rate = pickle.load(file)
    p0v = rate[0]
    p1v = rate[1]
    pAb = rate[2]
    return classifyNB(fileWordVec, p0v, p1v, pAb)


if __name__ == '__main__':
    rate = [0] * 100
    errorList, errorspam, errorham = [], [], []
    for i in range(0, 100):
        rate[i], test_error = spamTest()
        errorList = list(set(errorList) | set(test_error))
        # print('第{}次测试，朴素贝叶斯分类的错误率为：{}'.format(i + 1, rate[i]))  # 测试算法的错误率
    rate_average = np.mean(rate)
    print("共%d次测试，朴素贝叶斯分类平均错误率为：%f" % (i + 1, rate_average))
    for temp in errorList:
        if ((temp + 1) % 2 == 1):
            errorspam.append(int((temp + 3) / 2))
        else:
            errorham.append(int((temp + 1) / 2))
    print('预测错误的邮件为：垃圾邮件spam索引：{}   非垃圾邮件ham索引：{}'.format(errorspam, errorham))
    # filepath = input('输入需判断的邮件路径')
    # 判断某一路径下的邮件是否为垃圾邮件
    # if fileClassify('task1/spam/1.txt') == 1:
    #     print('垃圾邮件')
    # else:
    #     print('非垃圾邮件')
