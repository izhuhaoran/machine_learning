# -*- coding: utf-8 -*-

# Title : 使用决策树预测隐形眼镜类型
# Description :隐形眼镜数据是非常著名的数据集 ，它包含很多患者眼部状况的观察条件以及医生推荐的隐形眼镜类型 。
# 隐形眼镜类型包括硬材质 、软材质以及不适合佩戴隐形眼镜 。


from math import log
import operator
import matplotlib.pyplot as plt
from plot import plot_view
from treeStore import storeTree, grabTree
root = "0"

# 计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:     # 为所有可能分类创建字典
        currentLabel = featVec[-1]  # 取数据集的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0           # 分类标签值初始化
        labelCounts[currentLabel] += 1  # 给标签赋值
    shannonEnt = 0.0                    # 熵初始化
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries       # 求得每个标签的概率     # L(Xi) = -log2P(Xi)
        shannonEnt -= prob * log(prob, 2)   # 以2为底求对数      # H = - Σi=1 n  P(Xi)*log2P(Xi)
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
            reducedFeatVec.extend(featVec[axis+1:])
            # print("4",reducedFeatVec)
            retDataSet.append(reducedFeatVec)
            # print("5",retDataSet)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 计算特征的数目
    baseEntropy = calcShannonEnt(dataSet)   # 计算数据集的原始香农熵 用于与划分完的数据集的香农熵进行比较
    bestInfoGain = 0.0                      # 最佳信息增益初始化
    bestFeature = -1                        # 最佳划分特征初始化 TheBestFeatureToSplit
    for i in range(numFeatures):        # 遍历所有的特征
        featList = [example[i] for example in dataSet]  # 使用列表推导式创建列表 用于储存每一个数据的第i个特征
        # [ 表达式 for 变量 in 序列或迭代对象 ]             在这里的执行效果就是 每一列的特征都提取出来
        # aList = [ x ** 2 for x in range(10) ]
        # >>>aList  [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        uniqueVals = set(featList)       # 特征去重
        newEntropy = 0.0                 # 划分后信息熵初始化
        for value in uniqueVals:        # 遍历去重后的特征  分别计算每一个划分后的香农熵
            subDataSet = splitDataSet(dataSet, i, value)       # 划分
            prob = len(subDataSet)/float(len(dataSet))        # 算概率
            newEntropy += prob * calcShannonEnt(subDataSet)   # 算熵
        infoGain = baseEntropy - newEntropy     # 计算信息增益
        if (infoGain > bestInfoGain):       # 比较划分后的数据集的信息增益是否大于0 大于0 证明划分的有效
            bestInfoGain = infoGain         # 储存最佳信息增益值
            bestFeature = i                 # 储存最佳特征值索引
    return bestFeature                      # 返回最佳特征值索引


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
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
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
        return classList[0]     # 结束划分 如果只有一种分类属性  属性标签重复
    if len(dataSet[0]) == 1:    # 结束划分 如果没有更多的特征了  都为同一类属性标签了
        return majorityCnt(classList)   # 计数排序 取最大数特征
    bestFeat = chooseBestFeatureToSplit(dataSet)    # 获取最优特征索引
    bestFeatLabel = labels[bestFeat]                # 获取最优特征属性标签
    myTree = {bestFeatLabel: {}}                    # 决策树初始化 嵌套字典
    # print("0tree", myTree)
    del(labels[bestFeat])                           # 删除已经使用的特征标签 这时应只剩下有脚蹼特征了
    featValues = [example[bestFeat] for example in dataSet]     # 取出数据集所有最优属性值
    uniqueVals = set(featValues)                                 # 去重
    # print("标签%s,标签值%s" % (bestFeatLabel, uniqueVals))
    # 开始构建决策树
    for value in uniqueVals:
        subLabels = labels[:]   # 得到剩下的所有特征标签 作为我们的子节点可用
        # print("1tree", myTree)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree




def prodict():
    with open("task1\\lenses.txt", "rb") as fr:    # 取出数据集
        lenses = [inst.decode().strip().split('\t')for inst in fr.readlines()]
        # 解析由tap键分割的数据 去除数据左右的空格
        # 这里强调一下  决策树的数据集是由 特征属性值和分类标签两部分组成的
    lensesLabels = ['age', 'prescript', 'astigmatic', "tearRate"]    # 设置特征属性
    lensesTree = createTree(lenses, lensesLabels)                       # 创造样本决策树(分类器)
    return lensesTree  

"""
函数说明：
    对决策树进行分类
Parameters：
    inputTree：决策树
    featLabels：数据集中label顺序列表
    testVec：两个特征的属性值[特征一，特征二]
Rertun:
    classLabel：预测结果
    根据两个特征的属性值来预测分类
"""
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]    # 得到首key值
    secondDict = inputTree[firstStr]        # 首key的value--->下一个分支
    featIndex = featLabels.index(firstStr)  # 确定根节点是标签向量中的哪一个（索引）
    key = testVec[featIndex]                # 确定一个条件后的类别或进入下一个分支有待继续判别
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): # 判断实例函数 和 type函数类似 但是这个更好一点
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


if __name__ == "__main__":
    # mytree, labels = prodict()    这里返回的标签不完整 因为之前调用createTree（）时 取最优特征后删除了它在列表中的存在
    # 另外 只要一个返回值就删除另一个 不然只接受一个返回值
    # 计算机会将返回值变成元组类型 会在其他函数中需求取到列表的key值的时候 产生不必要的麻烦
    mytree = prodict()

    # storeTree(mytree, 'task1\\result\\tree_Storage.txt')
    # mytree = grabTree('task1\\result\\tree_Storage.txt')
    print(mytree)
    labels = ['age', 'prescript', 'astigmatic', "tearRate"]
    plot_view(mytree)
    result = classify(mytree, labels, ["presbyopic", "hyper", "yes", "normal"])
    if result == 'no lenses':
        print("视力良好")
    if result == 'soft':
        print("轻微近视")
    if result == 'hard':
        print("重度近视")
