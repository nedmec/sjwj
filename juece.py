import copy
from math import log
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'SimHei'  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


class DecisionTree(object):
    def __init__(self, decision_tree_type='CART', feature_list=None):
        self.decision_tree_type = decision_tree_type
        self.feature_list = feature_list

    @staticmethod
    def compute_entropy(x, y):
        """
        计算给定数据的信息熵 H(S) = -SUM(P*logP)
        :param x:
        :param y:
        :return:
        """
        sample_num = len(x)
        label_counter = Counter(y)
        dataset_entropy = 0.0
        for key in label_counter:
            prob = float(label_counter[key]) / sample_num
            dataset_entropy -= prob * log(prob, 2)  # get the log value

        return dataset_entropy

    def dataset_split_by_id3(self, x, y):
        """
        选择最好的数据集划分方式
        ID3算法：选择信息熵增益最大
        C4.5算法：选择信息熵增益比最大
        :param x:
        :param y:
        :return:
        """
        feature_num = len(x[0])
        base_entropy = self.compute_entropy(x, y)
        best_info_gain, best_info_gain_ratio = 0.0, 0.0
        best_feature_idx = -1
        for i in range(feature_num):
            unique_features = set([example[i] for example in x])
            new_entropy, split_entropy = 0.0, 0.0
            for feature in unique_features:
                sub_dataset, sub_labels = [], []
                for featVec, label in zip(x, y):
                    if featVec[i] == feature:
                        sub_dataset.append(list(featVec[:i]) + list(featVec[i + 1:]))
                        sub_labels.append(label)

                prob = len(sub_dataset) / float(len(x))
                new_entropy += prob * self.compute_entropy(sub_dataset, sub_labels)

            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_idx = i

        return best_feature_idx

    def dataset_split_by_c45(self, x, y):
        """
        选择最好的数据集划分方式
        C4.5算法：选择信息熵增益比最大
        :param x:
        :param y:
        :return:
        """
        feature_num = len(x[0])
        base_entropy = self.compute_entropy(x, y)
        best_info_gain, best_info_gain_ratio = 0.0, 0.0
        best_feature_idx = -1
        for i in range(feature_num):
            unique_features = set([example[i] for example in x])
            new_entropy, split_entropy = 0.0, 0.0
            for feature in unique_features:
                sub_dataset, sub_labels = [], []
                for featVec, label in zip(x, y):
                    if featVec[i] == feature:
                        sub_dataset.append(list(featVec[:i]) + list(featVec[i + 1:]))
                        sub_labels.append(label)

                prob = len(sub_dataset) / float(len(x))
                new_entropy += prob * self.compute_entropy(sub_dataset, sub_labels)

                split_entropy += -prob * log(prob, 2)

            info_gain = base_entropy - new_entropy
            info_gain_ratio = info_gain / split_entropy if split_entropy else 0.0
            if info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = info_gain_ratio
                best_feature_idx = i
        print("best_feature_idx", best_feature_idx)

        return best_feature_idx

    def create_tree_by_id3_and_c45(self, x, y, feature_list=None):
        """
        创建决策树
        :param x:
        :param y:
        :param feature_list:
        :return:
        """
        # the type is the same, so stop classify
        if len(set(y)) <= 1:
            return y[0]
        # traversal all the features and choose the most frequent feature
        if len(x[0]) == 1:
            return Counter(y).most_common(1)

        feature_list = [i for i in range(len(y))] if not feature_list else feature_list
        if self.decision_tree_type == 'ID3':
            best_feature_idx = self.dataset_split_by_id3(x, y)
        elif self.decision_tree_type == 'C45':
            best_feature_idx = self.dataset_split_by_c45(x, y)
        else:
            raise KeyError
        best_feature = feature_list[int(best_feature_idx)]  # 最佳特征
        decision_tree = {best_feature: {}}
        # feature_list = feature_list[:best_feature_idx] + feature_list[best_feature_idx + 1:]
        feature_list.pop(int(best_feature_idx))
        # get the list which attain the whole properties
        best_feature_values = set([sample[best_feature_idx] for sample in x])
        for value in best_feature_values:
            sub_dataset, sub_labels = [], []
            for featVec, label in zip(x, y):
                if featVec[best_feature_idx] == value:
                    sub_dataset.append(list(featVec[:best_feature_idx]) + list(featVec[best_feature_idx + 1:]))
                    sub_labels.append(label)

            decision_tree[best_feature][value] = self.create_tree_by_id3_and_c45(sub_dataset, sub_labels, feature_list)

        return decision_tree

    def train(self, x, y):
        x, y = np.array(x), np.array(y)
        if self.decision_tree_type in ('ID3', 'C45'):
            return self.create_tree_by_id3_and_c45(x, y, feature_list=copy.deepcopy(self.feature_list))

        else:
            raise KeyError

    def predict(self, tree, x):
        """
        决策树预测
        :param tree:
        :param x:
        :return:
        """
        root = list(tree.keys())[0]
        root_dict = tree[root]
        for key, value in root_dict.items():
            if x[root if not self.feature_list else self.feature_list.index(root)] == key:
                if isinstance(value, dict):
                    _label = self.predict(value, x)
                else:
                    _label = value

                return _label

        raise KeyError

    def getNumLeafs(self, myTree):
        numLeafs = 0
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if isinstance(secondDict[key],
                          dict):  # test to see if the nodes are dictonaires, if not they are leaf nodes
                numLeafs += self.getNumLeafs(secondDict[key])
            else:
                numLeafs += 1
        return numLeafs

    def getTreeDepth(self, myTree):
        maxDepth = 0
        firstStr = list(myTree.keys())[0]  # myTree.keys()[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if isinstance(secondDict[key],
                          dict):  # test to see if the nodes are dictonaires, if not they are leaf nodes
                thisDepth = 1 + self.getTreeDepth(secondDict[key])
            else:
                thisDepth = 1
            if thisDepth > maxDepth: maxDepth = thisDepth
        return maxDepth

    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                          xytext=centerPt, textcoords='axes fraction',
                          va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

    def plotTree(self, myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
        numLeafs = self.getNumLeafs(myTree)  # this determines the x width of this tree
        # depth = getTreeDepth(myTree)
        firstStr = list(myTree.keys())[0]  # myTree.keys()[0]     #the text label for this node should be this
        cntrPt = (self.xOff + (1.0 + float(numLeafs)) / 2.0 / self.totalW, self.yOff)
        self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(firstStr, cntrPt, parentPt, decisionNode)
        secondDict = myTree[firstStr]
        self.yOff = self.yOff - 1.0 / self.totalD
        for key in secondDict.keys():
            if isinstance(secondDict[key],
                          dict):  # test to see if the nodes are dictonaires, if not they are leaf nodes
                self.plotTree(secondDict[key], cntrPt, str(key))  # recursion
            else:  # it's a leaf node print the leaf node
                self.xOff = self.xOff + 1.0 / self.totalW
                self.plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, leafNode)
                self.plotMidText((self.xOff, self.yOff), cntrPt, str(key))
        self.yOff = self.yOff + 1.0 / self.totalD

    def createPlot(self, myTree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
        # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
        self.totalW = float(self.getNumLeafs(myTree))
        self.totalD = float(self.getTreeDepth(myTree))
        self.xOff = -0.5 / self.totalW
        self.yOff = 1.0
        self.plotTree(myTree, (0.5, 1.0), '')
        plt.show()


if __name__ == '__main__':
    X = [
        ["小", "否", "否", "一般"],
        ["小", "否", "否", "好"],
        ["小", "是", "否", "好"],
        ["小", "是", "是", "一般"],
        ["小", "否", "否", "一般"],
        ["中", "否", "否", "一般"],
        ["中", "否", "否", "好"],
        ["中", "是", "是", "好"],
        ["中", "否", "是", "非常好"],
        ["中", "否", "是", "非常好"],
        ["大", "否", "是", "非常好"],
        ["大", "否", "是", "好"],
        ["大", "是", "否", "好"],
        ["大", "是", "否", "非常好"],
        ["大", "否", "否", "一般"]
    ]
    Y = ["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"]

    X_test = [
        ["小", "否", "是", "一般"],
        ["中", "是", "否", "好"],
        ["大", "否", "是", "一般"],
    ]

    dt = DecisionTree(decision_tree_type='ID3', feature_list=['房', '车', '泳', '书'])
    myTree = dt.train(X, Y)
    print(myTree)
    # predict_label = dt.predict(myTree, X_test[1])
    # print(predict_label)
    # predict_label = dt.predict(myTree, X_test[2])
    # print(predict_label)
    # predict_label = dt.predict(myTree, X_test[1])
    # print(predict_label)
    # predict_label = dt.predict(myTree, X_test[2])
    # print(predict_label)
    dt.createPlot(myTree)
