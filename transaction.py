import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from collections import Counter

import time 
import sys
import os
from collections import Counter
import pickle as pkl
import pandas as pd
from matplotlib import pyplot as plt


def load_dataset(filename):
    dataset = pd.read_csv(filename,index_col = "TR_NO",usecols = ["TR_NO","TR_AMT","TR_BAL_AMT","FUND_USE","IS_CASH"])
    dataset = dataset.replace([np.inf, -np.inf], np.nan).fillna(value=0)
    return dataset

def get_negtive_trans(filename):
    return set(pd.read_csv(filename)["TR_NO"].values)


def creat_labels(training_dataset,cases):
    training_dataset["Y"] = 1
    training_dataset.loc[training_dataset.index.isin(cases),"Y"] = 0
    labels = training_dataset["Y"].values
    training_dataset.drop(labels = "Y",axis = 1,inplace = True)
    return labels


def create_features(dataset,FOUND_USE_DICT):
    def map_func(value):
        if value in FOUND_USE_DICT:
            return FOUND_USE_DICT[value]
        else:
            return -1

    dataset["FUND_USE"] = dataset["FUND_USE"].apply(map_func)
    table = dataset.groupby("TR_NO").size().to_dict()
    values = dataset.index.to_series().map(table)
    dataset["NUM_TRANS"] = values

    return dataset



# def filter_transaction(dataset):
#     filter_threshold = 5
#     filter_account = dict()
#     for name ,group in training_dataset.groupby("ACCT_ID"):
#         filter_account[name] = group.size

#     dataset["tm_count"] = 0

#     def func(row):
#         dataset["tm_count"] = filter_account[row["ACCT_ID"].values]
    
#     dataset.apply(func)

#     filter_dataset = dataset.loc[dataset["tm_count"] > 5,:]

#     return filter_dataset

                


def main(training_dataset,cases):
    labels = creat_labels(training_dataset,cases)

    X_train, X_test, y_train, y_test = train_test_split(training_dataset,labels,test_size=0.3, shuffle = False)
    # X_train = filter_transaction(X_train)

    FOND_USE_counter = Counter(X_train["FUND_USE"].values.tolist())
    FOUND_USE_DICT = {key:index for index,key in enumerate(FOND_USE_counter) if FOND_USE_counter[key] > 1}

    X_train = create_features(X_train,FOUND_USE_DICT)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = XGBClassifier(scale_pos_weight = Counter(y_train)[0] / Counter(y_train)[1])
    clf.fit(X_train, y_train)


    X_test = create_features(X_test,FOUND_USE_DICT)
    X_test = scaler.transform(X_test)

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))




if __name__ == '__main__':
    training_dataset_filename = "T3H_TRANS_ALL_WW_DATA_TABLE.csv"
    case_filename = "T3H_TRANS_YSB_WW_DATA_TABLE.csv"


    ## ------- load dataset --------
    if not os.path.exists("unsup_cases.pkl") or not os.path.exists("unsup_training_dataset.pkl"):
        training_dataset = load_dataset(training_dataset_filename)
        cases = get_negtive_trans(case_filename)

        with open("unsup_cases.pkl","wb") as f:
            pkl.dump(cases,f)
        with open("unsup_training_dataset.pkl","wb") as f:
            pkl.dump(training_dataset,f)
    else:
        with open("unsup_cases.pkl","rb") as f:
            cases = pkl.load(f)
        with open("unsup_training_dataset.pkl","rb") as f:
            training_dataset = pkl.load(f)      

    main(training_dataset, cases)




