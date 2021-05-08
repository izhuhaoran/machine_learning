# -*- coding: utf-8 -*-

# Title : 使用决策树预测隐形眼镜类型
# Description :隐形眼镜数据是非常著名的数据集 ，它包含很多患者眼部状况的观察条件以及医生推荐的隐形眼镜类型 。
# 隐形眼镜类型包括硬材质 、软材质以及不适合佩戴隐形眼镜 。

import pandas as pd
import numpy as np
from math import log
import operator
from plot import plot_view
from task2 import data_Get
from treeStore import storeTree, grabTree
root = "0"


# 计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # 为所有可能分类创建字典
        currentLabel = featVec[-1]  # 取数据集的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 分类标签值初始化
        labelCounts[currentLabel] += 1  # 给标签赋值
    shannonEnt = 0.0  # 熵初始化
    for key in labelCounts:
        prob = float(labelCounts[key]
                     ) / numEntries  # 求得每个标签的概率     # L(Xi) = -log2P(Xi)
        shannonEnt -= prob * log(
            prob, 2)  # 以2为底求对数      # H = - Σi=1 n  P(Xi)*log2P(Xi)
        # 注意这里是-= 虽然是求和 但是求和值<0 所以这里-=
    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    # (待划分的数据集、划分数据集的特征索引、特征的返回值)
    # 该函数是为了将划分的左右提取出来
    retDataSet = []
    for featVec in dataSet:
        # print("1",featVec)
        if featVec[axis] == value:
            # print("2",featVec[axis])
            reducedFeatVec = featVec[:axis]
            # print("3",reducedFeatVec)
            reducedFeatVec.extend(featVec[axis + 1:])
            # print("4",reducedFeatVec)
            retDataSet.append(reducedFeatVec)
            # print("5",retDataSet)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 计算特征的数目
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的原始香农熵 用于与划分完的数据集的香农熵进行比较
    bestInfoGain = 0.0  # 最佳信息增益初始化
    bestFeature = -1  # 最佳划分特征初始化 TheBestFeatureToSplit
    for i in range(numFeatures):  # 遍历所有的特征
        featList = [example[i]
                    for example in dataSet]  # 使用列表推导式创建列表 用于储存每一个数据的第i个特征
        # [ 表达式 for 变量 in 序列或迭代对象 ]             在这里的执行效果就是 每一列的特征都提取出来
        # aList = [ x ** 2 for x in range(10) ]
        # >>>aList  [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        uniqueVals = set(featList)  # 特征去重
        newEntropy = 0.0  # 划分后信息熵初始化
        for value in uniqueVals:  # 遍历去重后的特征  分别计算每一个划分后的香农熵
            subDataSet = splitDataSet(dataSet, i, value)  # 划分
            prob = len(subDataSet) / float(len(dataSet))  # 算概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 算熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if (infoGain > bestInfoGain):  # 比较划分后的数据集的信息增益是否大于0 大于0 证明划分的有效
            bestInfoGain = infoGain  # 储存最佳信息增益值
            bestFeature = i  # 储存最佳特征值索引
    return bestFeature  # 返回最佳特征值索引


"""
函数名称：majorityCnt（）
函数说明：统计classList中出现次数最多的元素（类标签）与K-近邻邻近K个元素排序函数功能一致
背景：如果数据集已经处理了所有属性，但是类标签依然不是唯一的
此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类。 
Parameters：
    classList：类标签列表
Returns：
    sortedClassCount[0][0]：出现次数最多的元素（类标签）
"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


"""
    函数名称：createTree（）
    函数说明：
        递归构建决策树 
        对算法步骤和具体递归赋值操作要多注意
    parameters:
        dataSet:数据集 
        labels:分类属性标签  
    returns：
        myTres：决策树 
"""


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # ["yes","yes"]
        return classList[0]  # 结束划分 如果只有一种分类属性  属性标签重复
    if len(dataSet[0]) == 1:  # 结束划分 如果没有更多的特征了  都为同一类属性标签了
        return majorityCnt(classList)  # 计数排序 取最大数特征
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 获取最优特征索引
    bestFeatLabel = labels[bestFeat]  # 获取最优特征属性标签
    myTree = {bestFeatLabel: {}}  # 决策树初始化 嵌套字典
    # print("0tree", myTree)
    del (labels[bestFeat])  # 删除已经使用的特征标签 这时应只剩下有脚蹼特征了
    featValues = [example[bestFeat] for example in dataSet]  # 取出数据集所有最优属性值
    uniqueVals = set(featValues)  # 去重
    # print("标签%s,标签值%s" % (bestFeatLabel, uniqueVals))
    # 开始构建决策树
    for value in uniqueVals:
        subLabels = labels[:]  # 得到剩下的所有特征标签 作为我们的子节点可用
        # print("1tree", myTree)
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def csv_data(dataset_Path):
    # dict_lenses = {}
    dataset, BSSID_List, fin_List= data_Get(dataset_Path)
    bssidlist = list(BSSID_List)
    lenses = [[0 for i in range(len(BSSID_List) + 1)]
              for i in range(len(fin_List))]

    for i in range(len(dataset)):
        times = dataset['finLabel'][i]
        bssid = dataset['BSSIDLabel'][i]
        room = dataset['RoomLabel'][i]

        bssid_index = bssidlist.index(bssid)
        lenses[times - 1][bssid_index] = 1
        lenses[times - 1][-1] = room
    # print(lenses)
    lensesLabels = bssidlist  # 设置特征属性

    text_save('task2/result/traindata.txt', lenses)
    text_save('task2/result/labels.txt', lensesLabels)
    # data_csv(lenses, lensesLabels)
    # lensesTree = createTree(lenses, lensesLabels)  # 创造样本决策树(分类器)
    return lenses, lensesLabels


def txt_tree():
    with open("task2\\result\\traindata.txt", "rb") as fr:  # 取出数据集
        lenses = [inst.decode().strip().split('\t') for inst in fr.readlines()]
    with open("task2\\result\\labels.txt", "rb") as fr:  # 取出数据集
        lensesLabels = [inst.decode().strip() for inst in fr.readlines()]
        # 解析由tap键分割的数据 去除数据左右的空格
        # 这里强调一下  决策树的数据集是由 特征属性值和分类标签两部分组成的
    lensesTree = createTree(lenses, lensesLabels)  # 创造样本决策树(分类器)
    return lensesTree, lensesLabels


def data_csv(lenses, labels):
    labels.append("Room")
    array = np.array(lenses)
    df = pd.DataFrame(array, columns=labels)
    df.to_csv('task2/result/traindata.csv')


def text_save(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '').replace(',', '\t')  # 去除[]
        s = s.replace("'", '').replace(',', '').replace(',', '\t') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")



def train_data(dataset_Path):
    # dict_lenses = {}
    dataset, BSSID_List, fin_List = data_Get(dataset_Path)
    bssidlist = list(BSSID_List)
    # finlist = list(fin_List)

    roomlist = [0 for i in range(len(fin_List))]
    lenses = [[0 for i in range(len(BSSID_List))]
              for i in range(len(fin_List))]

    for i in range(len(dataset)):
        times = dataset['finLabel'][i]
        bssid = dataset['BSSIDLabel'][i]
        room = dataset['RoomLabel'][i]

        bssid_index = bssidlist.index(bssid)
        lenses[times - 1][bssid_index] = 1
        roomlist[times - 1] = room
    # print(lenses)

    text_save('task2/result/testdata.txt', lenses)
    # text_save('task2/result/labels.txt', lensesLabels)
    # data_csv(lenses, lensesLabels)
    # lensesTree = createTree(lenses, lensesLabels)  # 创造样本决策树(分类器)
    return lenses, bssidlist, roomlist


def testtree(tree, dataset_Path):
    m = 0
    n = 0
    result_list = []
    lenses, lensesLabels, roomlist = train_data(dataset_Path)
    for testVec in lenses:
        result = classify(tree, lensesLabels, testVec)
        result_list.append(result)
    if len(result_list) == len(roomlist):
        for i in range(len(result_list)):
            if result_list[i] == roomlist[i]:
                m += 1
            else:
                n += 1
        result = float(m/(m+n))
        result = round(result, 2)
    else:
        print('验证结果生成数目错误')
        print(len(result_list))
        print(len(roomlist))
    text_save('task2/result/result.txt', result_list)
    text_save('task2/result/room.txt', roomlist)
    return result


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # 得到首key值
    secondDict = inputTree[firstStr]  # 首key的value--->下一个分支
    featIndex = featLabels.index(firstStr)  # 确定根节点是标签向量中的哪一个（索引）
    key = testVec[featIndex]  # 确定一个条件后的类别或进入下一个分支有待继续判别
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):  # 判断实例函数 和 type函数类似 但是这个更好一点
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

if __name__ == "__main__":
    file_path = 'task2\\TrainDT.csv'
    file2_path = 'task2\\TestDT.csv'
    
    filepath = 'task2/result/traindata.csv'

    lenses, labels = csv_data(file_path)   # 第一次初始读取原始csv文件处理数据
    mytree = createTree(lenses, labels)
    # mytree, labels = txt_tree()  # 读取处理好的数据txt文件生成树
    print(mytree)
    # storeTree(mytree, 'task2\\result\\tree_Storage.txt')  # 存储树
    # mytree = grabTree('task2\\result\\tree_Storage.txt')  # 读取树
    # print(mytree)
    rate = testtree(mytree, file2_path)
    print('测试集测试结果的正确率为：', rate)

    # plot_view(mytree) # 决策树可视化
    
    # result = classify(mytree, labels, ["presbyopic", "hyper", "yes", "normal"])
