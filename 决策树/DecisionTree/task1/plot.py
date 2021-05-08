import sys
from graphviz import Digraph
dot = Digraph(comment='The Test Table')
root = "0"


def treeplot(tree, node_num):
    global root
    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            dot.node(root, list(tree[first_label][i].keys())[0])
            dot.edge(node_num, root, str(i))
            treeplot(tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            dot.node(root, tree[first_label][i])
            dot.edge(node_num, root, str(i))



def getNumLeafs(myTree, father):
    global dot_num
    # global father
    numLeafs = 0        # 结点数目初始化
    temp_keys = list(myTree.keys()) 
    # mytree: {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    firstStr = temp_keys[0]     # 这里我们取到决策树的第一个key值
    if father == 0:
        dot.node(str(dot_num), firstStr)
        dot_num += 1

    secondDict = myTree[firstStr]   # 由于树的嵌套字典格式 我们通过第一个key得到了其value部分的另一个字典
    for key in secondDict.keys():   # 取出第二字典的key 0和1
        if type(secondDict[key]).__name__ == 'dict':
            # 判断是否相应key的value是不是字典 是字典就不是叶子结点
            # 继续调用本函数拆分该字典直到不是字典 即为叶子结点 进行记录
            temp_keys = list(secondDict[key].keys()) 
            firstStr = temp_keys[0] 
            edge = str(key)
            dot.node(str(dot_num), firstStr)
            dot.edge(str(father), str(dot_num), edge)
            father = dot_num
            dot_num += 1

            print(dot.source)
            sys.stdout.flush()

            numLeafs += getNumLeafs(secondDict[key], father)
        else:   # 不是字典直接记录为叶子结点
            firstStr = secondDict[key]
            edge = str(key)
            dot.node(str(dot_num), firstStr)
            dot.edge(str(father), str(dot_num), edge)
            dot_num += 1

            print(dot.source)
            sys.stdout.flush()

            numLeafs += 1
    print(dot.source) 
    sys.stdout.flush()
    return numLeafs


def plot_view(myTree):
    first_label = list(myTree.keys())[0]
    dot.node("0", first_label)
    treeplot(myTree, "0")
    dot.render('task1/result/mytree.gv', view=True)


if __name__ == "__main__":
    myTree = {'tearRate': {'normal': {'astigmatic': {'no': {'age': {'presbyopic': {'prescript': {'hyper': 'soft', 'myope': 'no lenses'}}, 'pre': 'soft', 'young': 'soft'}}, 'yes': {'prescript': {'hyper': {'age': {'presbyopic': 'no lenses', 'pre': 'no lenses', 'young': 'hard'}}, 'myope': 'hard'}}}}, 'reduced': 'no lenses'}}
    plot_view(myTree)