#!/usr/bin/env python3

# info
# -name   : zhangruochi
# -email  : zrc720@gmail.com



import pandas as pd
import numpy as np 
import pickle as pkl
import random
#from matplotlib import pyplot as plt

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
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import os


#from sklearn.preprocessing import OneHotEncoder


def get_days_diff(grouped):
    def row_func(diff):
        return diff.days+1

    timeseries = grouped["TR_TM"].max() - grouped["TR_TM"].min()
    timeseries = timeseries.apply(row_func)
    return timeseries



def transform_feature(dataset):
    grouped = dataset.groupby("ACCT_ID")
    
    data_dict = {
                 "ACCT_COUNT": grouped["ACCT_ID"].count(),
                 "TR_SUM": grouped["TR_AMT"].sum(),
                 "TR_MEAN": grouped["TR_AMT"].mean(), 
                 "TR_STD":grouped["TR_AMT"].std(),
                 "TR_COUNT": grouped["TR_AMT"].count(),
                 "BAL_PERCENT": grouped["TR_BAL_AMT"].mean()/grouped["TR_AMT"].mean(), 
                 # avg of transaction count of 1 days
                 "TR_FREQ": grouped["TR_AMT"].count()/ get_days_diff(grouped)
    }

    transformed_dataset = pd.DataFrame(data_dict)
    transformed_dataset = transformed_dataset.replace([np.inf, -np.inf], np.nan).fillna(value=0).astype(np.float64)

    return transformed_dataset




def load_dataset(filename):
    dataset = pd.read_csv(filename,low_memory=False)
    valid_dataset = dataset.loc[:,["TR_TM","ACCT_ID","TR_AMT","TR_BAL_AMT"]]
    valid_dataset["TR_TM"] = valid_dataset["TR_TM"].apply(pd.to_datetime)

    return valid_dataset


def normalize(dataset):
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataset)



def get_negtive_customer(filename):
    return set(pd.read_csv(filename)["ACCT_ID"].values)
    


def create_trainset_label(filtered_positive_dataset, negetive_dataset):
    labels = np.array([1] * filtered_positive_dataset.shape[0] + [0] * negetive_dataset.shape[0])
    filtered_positive_dataset = filtered_positive_dataset.append(negetive_dataset)
    filtered_positive_dataset.fillna(method = "backfill",inplace = True)
    #filtered_positive_dataset = normalize(filtered_positive_dataset)

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
        estimator = RandomForestClassifier()    

    return estimator


# 采用 K-Fold 交叉验证 得到 aac

def get_acc(estimator, X, y, skf):
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        scores.append(estimator.score(X_test, y_test))

    return np.mean(scores) 


def training(training_dataset,labels):
    skf = StratifiedKFold(n_splits=10)
    estimator_list = [5]
    seed_number = 10

    max_estimator_aac = -1
    estimator_name = ["SVM","KNN","DT","NB","LG","RF"]

    for index in estimator_list:
        estimator_acc = get_acc(select_estimator(
            index), training_dataset, labels,skf)
        print("{} accuracy: {} ".format(estimator_name[index],estimator_acc))
        if estimator_acc > max_estimator_aac:
            max_estimator_aac = estimator_acc  # 记录对于 k 个 特征 用四个estimator 得到的最大值

    print(max_estimator_aac)
    


## to balance the positive dataset and negative dataset, filter part of positive dataset
def filter_positive_dataset(positive_dataset,multiple,num_of_negetive):
    sample_list = random.sample(list(range(positive_dataset.shape[0])), num_of_negetive*multiple)
    return positive_dataset.iloc[sample_list,:]


def filter_negetive_dataset(negetive_dataset,multiple):
    
    for i in range(multiple):
        negetive_dataset = negetive_dataset.append(negetive_dataset,ignore_index = True)

    return negetive_dataset




def plot_roc(auc_value,fpr,tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Transcation Suspicious Activities Detection')
    plt.legend(loc="lower right")
    plt.show()




def tune_parameters(training_dataset,labels):
    """
    tuned_parameters = {'kernel': ["rbf"], 
                        'gamma': [1e-2,1e-3,1e-4],
                        'C': [1,10] }
    """ 
    """
    tuned_parameters = {'n_neighbors': [3,5,10],
                        'weights':['uniform','distance'], 
                        'algorithm': ["ball_tree", "kd_tree", "brute"],
                        'leaf_size': [20,30,50] }                   
    """

    tuned_parameters = {
        "n_estimators": [200],
        "criterion": ["gini"],
        "min_samples_leaf": [50],
        "n_jobs": [-1],
    }


    scores = ['roc_auc']

    # 将数据集分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
    training_dataset, labels, test_size=0.3, random_state=0)
                        
    for score in scores:
        print("# Tuning hyper-parameters for %s\n" % score)

        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv = 5,
                           scoring='%s' % score)

        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:\n")
        #输出最优的模型参数
        print(clf.best_params_)
        with open("model.pkl","wb") as f:
            pkl.dump(clf,f)
        print()
        print("Detailed classification report:\n")
        #在测试集上测试最优的模型的泛化能力.

        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        
        # print("plot the orc curve:\n")
        # y_pred_pro = clf.predict_proba(X_test)
        # y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values
        # auc_value = roc_auc_score(y_true, y_scores)
        # fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
        # plot_roc(auc_value,fpr,tpr)
        
    return clf

def predict(clf,dataset_filename):
    """ 
        input: supervise learning model
        output: predict result, in form of dict, {account_number: positive(negetive)}
    """
    ans = {}
    if os.path.exists("test_set.pkl"):
        with open("test_set.pkl","rb") as f:
            transform_dataset = pkl.load(f)
    else:        
        training_dataset = load_dataset(dataset_filename)
        transform_dataset = transform_feature(training_dataset)
        with open("test_set.pkl","wb") as f:
            pkl.dump(transform_dataset,f)
    
    print("predict dataset shape: " + str(transform_dataset.shape))
    label = clf.predict(transform_dataset)

    for index,name in enumerate(transform_dataset.index):
        ans[name] = label[index]

    return ans,transform_dataset,label


def get_model(training_dataset,cases):
    """
    input:
       training_dataset: 所有的反洗钱数据
       cases: 反洗钱软件产生案件的数据(或者上报的数据)
    output:
        supervise learning model
    """

    transform_dataset = transform_feature(training_dataset)
    positive_customer = set(transform_dataset.index.values).difference(cases)

    negetive_dataset = transform_dataset.iloc[transform_dataset.index.isin(cases),:]
    positive_dataset = transform_dataset.iloc[transform_dataset.index.isin(positive_customer),:]


    negetive_dataset,test_negetive_dataset = negetive_dataset.iloc[:int(negetive_dataset.shape[0]*0.8),:],negetive_dataset.iloc[int(negetive_dataset.shape[0]*0.8):,:]
    positive_dataset,test_positive_dataset = positive_dataset.iloc[:int(positive_dataset.shape[0]*0.8),:],positive_dataset.iloc[int(positive_dataset.shape[0]*0.8):,:]


    #filtered_positive_dataset = filter_positive_dataset(positive_dataset,5,len(cases))
    
    negetive_dataset = filter_negetive_dataset(negetive_dataset,13)
    print("negetive_dataset: " + str(negetive_dataset.shape))


    training_dataset,labels = create_trainset_label(positive_dataset,negetive_dataset)
    print("final training dataset:" + str(training_dataset.shape))


    test_negetive_dataset = filter_negetive_dataset(test_negetive_dataset,13)
    test_dataset,test_labels = create_trainset_label(test_positive_dataset,test_negetive_dataset)
    print("final testing dataset:" + str(test_dataset.shape))

    """
    print("------------------\n")
    print("dataset and labels for training: ")
    print("dataset: {}".format(training_dataset.shape))
    print("labls: {}".format(len(labels)))    
    """

    # print("------------------\n")
    # print("training: ......")
    # training(training_dataset,labels)


    print("------------------\n")
    print("tune parameters: ......")

    clf = tune_parameters(training_dataset,labels)

    print("testing result.....")

    test_pred = clf.predict(test_dataset)
    print(classification_report(test_labels, test_pred))


    return clf


def score_for_account(predicted_result,dataset):

    error_account = [key for key in predicted_result if not predicted_result[key]]

    error_dataset = dataset.iloc[dataset.index.isin(error_account),:]

    tmp1 = abs(error_dataset["TR_MEAN"] - np.mean(error_dataset["TR_MEAN"])) / np.std(error_dataset["TR_STD"])
    tmp2 = abs(error_dataset["TR_COUNT"] - np.mean(error_dataset["TR_COUNT"])) / np.std(error_dataset["TR_COUNT"])
    score = 2*(tmp1*tmp2) / (tmp1+tmp2)
    normalize_score = (score - np.min(score)) / (np.max(score) - np.min(score))

    return normalize_score



def get_true_label(dataset,cases):
    labels = []
    def func(row):
        if row.name in cases:
            labels.append(0)
        else:
            labels.append(1)
    dataset.apply(func,axis = 1)

    return labels


if __name__ == '__main__':

    training_dataset_filename = "T3H_TRANS_ALL_WW_DATA_TABLE.csv"
    case_filename = "T3H_TRANS_YSB_WW_DATA_TABLE.csv"
    new_dataset = "T3H_TRANS_ALL_WW_DATA_TABLE.csv"



    if not os.path.exists("cases.pkl") or not os.path.exists("training_dataset.pkl"):

        ## ------- 数据读取接口 --------
        training_dataset = load_dataset(training_dataset_filename)
        cases = get_negtive_customer(case_filename)


        with open("cases.pkl","wb") as f:
            pkl.dump(cases,f)
        with open("training_dataset.pkl","wb") as f:
            pkl.dump(training_dataset,f)
    else:
        with open("cases.pkl","rb") as f:
            cases = pkl.load(f)
        with open("training_dataset.pkl","rb") as f:
            training_dataset = pkl.load(f)

        print("abnormal account: " + str(len(cases)))
        print("raw training dataset shape: " + str(training_dataset.shape))

    ## ------- model -------------
    clf = get_model(training_dataset,cases)
    exit()
    predicted_result,dataset,y_pred = predict(clf,new_dataset)

    y_true = get_true_label(dataset,cases)

    print(sum(y_true) - sum(y_pred))

    print(classification_report(y_true, y_pred))
    


    with open("predicted_label.pkl","wb") as f:
        pkl.dump(predicted_result,f)

    scores = score_for_account(predicted_result,dataset)

    with open("supervised_result.pkl","wb") as f:
        pkl.dump(scores,f)
