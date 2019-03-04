**This is the Unsupervise Machine Learning Model which trains the dataset and then gives all the transaction a abnormal score. A parameter of thereshold can be given to predict whether a transaction is normal or not.**

## Language and Framework
1. **Language**
    - Python 3.7.0
2. **Frameword**
    - pandas==0.23.4
    - numpy==1.15.2
    - scikit-learn==0.20.0
    - matplotlib==3.0.0
    - scipy==1.1.0


## API
1. **Dataset** (暂时用的csv文件，到时候根据实际需求更改)
    - load_dataset()
        - input: dataset filename
        - ouput: pandas dataset, relation of transaction and teller account.
    - get_negtive_customer():
        - input:  case_filename
        - output: a set of negetive account name

2. **Model training**
    - feature: 如果效果不好，再扩充特征。
    - get_model()
        - input: training_dataset,cases, threshold(异常值低于threshold会被判定为abnormal)
        - output: model

3. **Predict**
    - predict()
        - input: model, dataset
        - output: predicted label, as a dict of account name and it's label

4. **abnormal_score**
    - get_abnormal_score()
        - input: model, dataset
        - output: predicted abnormal score, as a dict of account name and it's abnormal score
          **The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.**

4. **Score for account**
    - score_for_account()
        - input: abnormal socre for every transaction, dataset for predicting(the out put of abnormal_score function)
        - output: the score for account    

## Algorithm
1. **Loading dataset**
    - transform the dataset to the DataFrame

2. **Feature transforming** 
    - use all the feature of transaction

3. **Using the Isolation Forest to train the dataset, and classifier the dataset to normal and abnormal**
    - Return the anomaly score of each sample

    The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
    Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
    This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
    Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.


4. **Using best model with best prameters to predict new dataset**




## Scoring

1. 在知识图谱中标注柜员操作的所有交易。
2. 对其所有交易的异常值分数求平均值，作为柜员的异常交易分数。





    
    