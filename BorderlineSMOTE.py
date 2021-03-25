import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import BaggingClassifier


# 1.读取数据集
path = 'E:/csv2/lbp11/train5_a6.csv'
train1 = np.loadtxt(path, dtype=float, delimiter=',')
print(train1.shape)

path = 'E:/csv2/lbp11/val5_a6.csv'
val1 = np.loadtxt(path, dtype=float, delimiter=',')
print(val1.shape)

path = 'E:/csv3/2D-lbp/test5_LBP1D.csv'
test1 = np.loadtxt(path, dtype=float, delimiter=',')
print(test1.shape)

data_train = np.vstack((train1, val1))
print(data_train.shape)

# 2.划分数据与标签
x_train, y_train = np.split(data_train, indices_or_sections=(7680*2,), axis=1)  # x为数据，y为标签
x_test, y_test = np.split(test1, indices_or_sections=(7680*2,), axis=1)  # x为数据，y为标签

x_train = x_train[:, ::2]
x_test = x_test[:, ::2]

nca = NeighborhoodComponentsAnalysis(random_state=42, n_components=100, init='pca')


std = MinMaxScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)


nca.fit(x_train, y_train)
x_train1 = nca.transform(x_train)
x_test1 = nca.transform(x_test)


# imbalance
sm = BorderlineSMOTE(random_state=0)
x_train1, y_train = sm.fit_resample(x_train1, y_train)


# # 3.训练KNN分类器
# C_list = [1]
# x = [['kernel', 'c', 'gamma', 'acc']]
# for C_index in range(len(C_list)):
#     classifier = KNeighborsClassifier(n_neighbors=C_list[C_index])
#     classifier.fit(x_train1, y_train.ravel())
#     score = balanced_accuracy_score(y_test, classifier.predict(x_test1))
#     aaa = [C_list[C_index], score]
#     print(aaa)


# 3.训练svm分类器
C_list = [40]
kernel_list = ['rbf']
x = [['kernel', 'c', 'gamma', 'acc']]
for kernel_index in range(len(kernel_list)):
    for C_index in range(len(C_list)):
        classifier = svm.SVC(C=C_list[C_index], kernel=kernel_list[kernel_index], probability=True)
        classifier.fit(x_train1, y_train.ravel())
        score = balanced_accuracy_score(y_test, classifier.predict(x_test1))
        aaa = [kernel_list[kernel_index], C_list[C_index], score]
        print(aaa)


