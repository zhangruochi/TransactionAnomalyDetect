**This is the Supervise Machine Learning Model to predict whether a account_id is normal or not. If a account_id is abnormal, the model will give a abnormal score to it.**

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
        - ouput: pandas dataset
    - get_negtive_customer():
        - input:  case_filename
        - output: a set of negetive account name

2. **Model training**
    - get_model()
        - input: training_dataset,cases
        - output: model

3. **Predict**
    - predict()
        - input: model, dataset
        - output: predicted label, as a dict of account name and it's label

4. **Score for account**
    - score_for_account()
        - input: predicted label of account, dataset for predicting(the out put of predict function)
        - output: the score for abnormal account
        

## Algorithm
1. **Loading dataset**
    - transform the dataset to the DataFrame

2. **Feature transforming** 
    - "CUST_COUNT":  每个 account 包含的 customer_id 的数量
    - "TR_SUM": 每个 account 的交易总数
    - "TR_MEAN": 每个 account 的交易平均值
    - "TR_STD": 每个 account 的交易标准差
    - "TR_COUNT": 每个 account 的交易数量
    - "BAL_PERCENT": 每个 account 剩余钱数的平均值/ 每个 account 交易的总钱数
    - "TR_FREQ": 每个 account 以天为单位的交易频率

3. **Comparing five machine learning model and choose the best model**
    - SVM
    - KNN
    - NB
    - LG
    - RF

4. **Choosing the best model to tune the prameters, and renew the best model**
    - using the roc as the objective of optimazation.

5. **using best model with best prameters to predict new dataset**


## Scoring

### amount of money
1. 计算账户i的平均交易金额:  **x(i)**
2. 计算所有账户的平均交易金额: **u**
3. 计算所有账户的交易金额的标准差: **σ**
4. 对所有x(i), 求其标准分数: 

> z(i) = (x(i) – μ) / σ

5. 账号的异常值为:

> s1(i) = z(i) / Max(z)

### the number of transaction
1. 计算账户i的交易总数: **x(i)**
2. 计算所有账户的平均交易数: **u**
3. 计算所有账户的交易数的标准差: **σ**
4. 对所有x(i), 求其标准分数:

> s2(i) = (x(i) – μ) / σ

5. 账号的异常值为:

> s2(i) = z(i) / Max(z)


### combine the number of transaction and amount of money

> score(i) = 2*(s1(i) * s2(i))/(s1(i) + s2(i))
  normalize_score(i) = (score(i) - min(score(i))) / max(score(i)) - min(score(i))


### Conclusion
1. 对于柜员的银行账户，上述 score(i) 即为其异常值分数
2. 对于柜员经办的异常账户，在知识图谱中标注柜员所有能够直接相连的异常账户的异常值的平均数。



    
    