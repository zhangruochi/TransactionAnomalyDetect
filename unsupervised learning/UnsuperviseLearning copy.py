#!/usr/bin/env python3
# info
# -name   : zhangruochi
# -email  : zrc720@gmail.com
import pandas as pd
import numpy as np
import pickle as pkl
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind_from_stats
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#from sklearn.preprocessing import OneHotEncoder
def get_days_diff(grouped):
    def row_func(diff):
        return diff.days+1
    timeseries = grouped["TR_TM"].max() - grouped["TR_TM"].min()
    timeseries = timeseries.apply(row_func)
    return timeseries
def transform_feature(dataset):
    grouped = dataset.groupby("CUST_ID")

    data_dict = {"TR_SUM": grouped["TR_AMT"].sum(),
                 "TR_MEAN": grouped["TR_AMT"].mean(),
                 "TR_STD":grouped["TR_AMT"].std(),
                 "TR_COUNT": grouped["TR_AMT"].count(),
                 "BAL_PERCENT": grouped["TR_BAL_AMT"].mean()/grouped["TR_AMT"].mean(),
                 # avg of transaction count of 1 days
                 "TR_FREQ": grouped["TR_AMT"].count()/ get_days_diff(grouped)
                 }
    transformed_dataset = pd.DataFrame(data_dict)
    print("dataset for training: " + str(transformed_dataset.shape))
    return transformed_dataset

def load_dataset(filename):
    dataset = pd.read_csv(filename)
    valid_dataset = dataset.loc[:,["TR_TM","CUST_ID","TR_AMT","TR_BAL_AMT",]]
    valid_dataset["TR_TM"] = valid_dataset["TR_TM"].apply(pd.to_datetime)
    return valid_dataset
def get_negtive_customer(filename):
    return set(pd.read_csv(filename)["CUST_ID"].values)

def normalize(dataset):
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataset)
def create_trainset_label(filtered_positive_dataset, negetive_dataset):
    labels = np.array([1] * filtered_positive_dataset.shape[0] + [0] * negetive_dataset.shape[0])
    filtered_positive_dataset = filtered_positive_dataset.append(negetive_dataset)
    filtered_positive_dataset.fillna(method = "backfill",inplace = True)
    filtered_positive_dataset = normalize(filtered_positive_dataset)
    return filtered_positive_dataset,labels

# 选择分类器 D-tree,SVM,NBayes,KNN
def select_estimator(case):
    if case == 0:
        estimator = SVC(gamma = "auto")
    elif case == 1:
        estimator = KNeighborsClassifier()
    elif case == 2:
        estimator = DecisionTreeClassifier(random_state=7)
    elif case == 3:
        estimator = GaussianNB()
    elif case == 4:
        estimator = LogisticRegression(solver = 'liblinear')
    elif case == 5:
        estimator = RandomForestClassifier(n_estimators = 100)
    return estimator
# 采用 K-Fold 交叉验证 得到 aac
def get_acc(estimator, X, y, skf):
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train)
        scores.append(estimator.score(X_test, y_test))
    return np.mean(scores)
def training(X,y):
    skf = StratifiedKFold(n_splits=10)
    estimator = IsolationForest(contamination = 0.1,behaviour="new")
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train)
        scores.append(np.mean(estimator.predict(X_test) ==  y_test))
    return np.mean(scores)
def filter_positive_dataset(positive_dataset,multiple,num_of_negetive):
    sample_list = random.sample(list(range(positive_dataset.shape[0])), num_of_negetive*multiple)   # 重启的组数为所有特征的一半
    return positive_dataset.iloc[sample_list,:]
def fit_and_plot_(training_dataset,labels):
    # 将数据集分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
    training_dataset, labels, test_size=0.3, random_state=0)
    clf = IsolationForest(contamination = 0.1,behaviour="new")
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print("training dataset predict score: {}".format(np.mean(y_pred_train == y_train)))
    print("testing dataset predict score : {}\n".format(np.mean(y_pred_test == y_test)))
    #c1 = ["green" if label else "red" for label in y_train == y_pred_train ]
    c2 = ["green" if label else "red" for label in y_test == y_pred_test ]
    #b1 = plt.scatter(X_train[:,2], X_train[:,3], c=c1,
    #                 s=20, edgecolor='k',label="training observations")
    b2 = plt.scatter(X_test[:,3], X_test[:,5], c=c2,
                     s=20, edgecolor='k',label = "testing observations")
    plt.axis('tight')
    plt.xlim((0, 0.01))
    plt.ylim((0, 0.01))
    #plt.legend(loc="upper left")
    plt.show()
def main():
    """
    training_dataset_filename = "T3H_TRANS_ALL_WW_DATA_TABLE.csv"
    training_dataset = load_dataset(training_dataset_filename)
    transform_dataset = transform_feature(valid_dataset)
    report_filename = "T3H_TRANS_YSB_WW_DATA_TABLE.csv"
    case_filename = "T3H_TRANS_MODEL_WW_DATA_TABLE.csv"

    report_negetive_customer = get_negtive_customer(report_filename)
    case_negetive_customer = get_negtive_customer(case_filename)
    with open("training_dataset.pkl","wb") as f:
        pkl.dump(training_dataset,f)
    with open("report_negetive_customer.pkl","wb") as f:
        pkl.dump(report_negetive_customer,f)
    with open("case_negetive_customer.pkl","wb") as f:
        pkl.dump(case_negetive_customer,f)
    with open("transform_dataset.pkl","wb") as f:
        pkl.dump(transform_dataset,f)
    """


    with open("training_dataset.pkl","rb") as f:
        valid_dataset = pkl.load(f)
    with open("report_negetive_customer.pkl","rb") as f:
        report_negetive_customer = pkl.load(f)

    with open("case_negetive_customer.pkl","rb") as f:
        case_negetive_customer = pkl.load(f)

    with open("transform_dataset.pkl","rb") as f:
        transform_dataset = pkl.load(f)
    print("raw dataset: " + str(valid_dataset.shape))
    print("transformed dataset: " + str(transform_dataset.shape))
    print("report dataset numbers: {}".format(len(report_negetive_customer)))
    print("case dataset numbers: {}".format(len(case_negetive_customer)))
    print("------------------\n")
    positive_customer = set(transform_dataset.index.values).difference(case_negetive_customer)
    negetive_dataset = transform_dataset.iloc[transform_dataset.index.isin(case_negetive_customer),:]
    positive_dataset = transform_dataset.iloc[transform_dataset.index.isin(positive_customer),:]
    filtered_positive_dataset = filter_positive_dataset(positive_dataset,5,len(case_negetive_customer))
    print("negetive dataset: \n")
    print(negetive_dataset.iloc[:5,:])
    print("positive_dataset: \n")
    print(filtered_positive_dataset.iloc[:5,:])
    training_dataset,labels = create_trainset_label(filtered_positive_dataset,negetive_dataset)

    print("------------------\n")
    print("dataset and labels for training: ")
    print("dataset: {}".format(training_dataset.shape))
    print("labls: {}\n".format(len(labels)))
    print("------------------\n")
    print("training ......")
    scores = training(training_dataset,labels)
    print("10 fold average accuracy for IsolationForest: {}".format(scores))
    print("------------------\n")
    print("training and plot ......")
    fit_and_plot_(training_dataset,labels)



if __name__ == '__main__':
    main()
