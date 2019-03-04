#!/usr/bin/env python3
# info
# -name   : zhangruochi
# -email  : zrc720@gmail.com
import pandas as pd
import numpy as np 
import pickle as pkl
import random
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
import os
#from sklearn.preprocessing import OneHotEncoder
def load_dataset(filename):
    dataset = pd.read_csv(filename,index_col = "TR_NO")
    dataset = dataset.iloc[~dataset.index.duplicated(keep="first")]

    valid_dataset = dataset.loc[:,["TR_CHNL","TR_AMT","TR_BAL_AMT","CUST_TYPE","TR_CD","IS_CASH","DEBIT_CREDIT", "CURR_CD", "TR_IPV4"]]
    valid_dataset = pd.get_dummies(valid_dataset, dummy_na=True)
    
    rela_table = {}
    def rela(row):
        rela_table[row.name] = row["OPR_ID"]
    dataset.apply(func = rela,axis = 1)
    
    valid_dataset = valid_dataset.replace([np.inf, -np.inf], np.nan).fillna(value=0).astype(np.float64)

    return valid_dataset,rela_table


def get_negtive_customer(filename):
    return set(pd.read_csv(filename)["TR_NO"].values)


def normalize(dataset):
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataset)

def create_trainset_label(filtered_positive_dataset, negetive_dataset):
    labels = np.array([1] * filtered_positive_dataset.shape[0] + [-1] * negetive_dataset.shape[0])
    training_dataset = filtered_positive_dataset.append(negetive_dataset)    
    #training_dataset  = normalize(training_dataset)
    return training_dataset ,labels
    
def training(X,y,threshold):
    skf = StratifiedKFold(n_splits=10)
    estimator = IsolationForest(contamination = threshold,behaviour="new")
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train)
        scores.append(np.mean(estimator.predict(X_test) ==  y_test))
    return np.mean(scores)

def filter_positive_dataset(positive_dataset,multiple,num_of_negetive):
    sample_list = random.sample(list(range(positive_dataset.shape[0])), num_of_negetive*multiple)  
    return positive_dataset.iloc[sample_list,:]

def fit_and_plot(training_dataset,labels):
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

def get_model(training_dataset,cases,threshold):
    """
    input:
       training_dataset: 所有的反洗钱数据
       cases: 反洗钱软件产生案件的数据(或者上报的数据)
    output:
        unsupervise learning model
    """
    positive_trans = set(training_dataset.index.values).difference(cases)
    negetive_dataset = training_dataset.iloc[training_dataset.index.isin(cases),:]
    positive_dataset = training_dataset.iloc[training_dataset.index.isin(positive_trans),:]
    #filtered_positive_dataset = filter_positive_dataset(positive_dataset,1,len(cases))
    training_dataset,labels = create_trainset_label(positive_dataset,negetive_dataset)

    
    #fit_and_plot(training_dataset,labels)
    
    clf = IsolationForest(contamination = threshold, behaviour="new")
    clf.fit(training_dataset)
    return clf,labels
    
def predict(clf,dataset_filename):
    """ 
        input: unsupervise learning model
        output: predict result, in form of dict, {transaction: positive(negetive)}
    """
    ans = {}
    training_dataset,_ = load_dataset(dataset_filename)
    label = clf.predict(training_dataset)

    return label

def get_abnormal_score(clf,dataset_filename):
    """ 
        input: unsupervise learning model
        output: predict result, in form of dict, {transaction: abnormal_score}
    """
    ans = {}
    training_dataset,_ = load_dataset(dataset_filename)
    score = clf.decision_function(training_dataset)
    for index,name in enumerate(training_dataset.index):
        ans[name] = abs(score[index])
    return ans,training_dataset
    
    
def score_for_account(abnormal_score,dataset,rela_table):
    dataset["score"] = 0
    dataset["operator"] = ""
        
    for name in dataset.index:
        dataset.loc[name,"score"] = abnormal_score[name]
        dataset.loc[name,"operator"] = rela_table[name]
    score = dataset.groupby("operator")["score"].mean().to_dict()
    return score
                        
if __name__ == '__main__':
    training_dataset_filename = "T3H_TRANS_ALL_WW_DATA_TABLE.csv"
    case_filename = "T3H_TRANS_YSB_WW_DATA_TABLE.csv"
    new_dataset = "T3H_TRANS_ALL_WW_DATA_TABLE.csv"
    
    ## ------- load dataset --------
    if not os.path.exists("cases.pkl") or not os.path.exists("training_dataset.pkl") or not os.path.exists("rela_table"):
        training_dataset,rela_table = load_dataset(training_dataset_filename)
        cases = get_negtive_customer(case_filename)

        with open("cases.pkl","wb") as f:
            pkl.dump(cases,f)
        with open("training_dataset.pkl","wb") as f:
            pkl.dump(training_dataset,f)
        with open("rela_table.pkl","wb") as f:
            pkl.dump(rela_table,f)
    else:
        with open("cases.pkl","rb") as f:
            cases = pkl.load(f)
        with open("training_dataset.pkl","rb") as f:
            training_dataset = pkl.load(f)
        with open("rela_table.pkl","rb") as f:
            rela_table = pkl.load(f)

    ## ------- model --------------
    clf,y_true = get_model(training_dataset,cases,threshold = len(cases)/training_dataset.shape[0])
    ## ------- predict ------------
    y_pred = predict(clf,new_dataset)

    print(classification_report(y_true, y_pred)) 
    exit()

    abnormal_score,dataset = get_abnormal_score(clf,new_dataset)
    score = score_for_account(abnormal_score,dataset,rela_table)