import pandas as pd
import numpy as np
import pickle as pkl
import csv



def load_dataset(filename):
    dataset = pd.read_csv(filename,low_memory=False)
    valid_dataset = dataset.loc[:,["TR_TM","CUST_ID","ACCT_ID","TR_AMT","TR_BAL_AMT",]]
    valid_dataset["TR_TM"] = valid_dataset["TR_TM"].apply(pd.to_datetime)

    return valid_dataset


def create_account():
    account_set = set()
    flag = True
    with open("T3H_TRANS_YSB_WW_DATA_TABLE.csv",mode = "r",encoding='utf-8', errors='ignore' ) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if flag:
                flag = False
                continue
            account_set.add(line[4])    
            account_set.add(line[13])
    print(len(account_set))
    exit()
    with open("account.csv","w") as csvfile:
        writer = csv.writer(csvfile)       
        writer.writerow(["","ACCT_ID"])
        index = 0
        for account in list(account_set):
            if not account:
                continue
            writer.writerow([index,str(account)])
            index += 1



def create_hall():
    filename = "F_CM_KTLP_GYCSHU_DATA_TABLE.csv"
    dataset = pd.read_csv(filename)
    valid_dataset = dataset.loc[:,"YNGYJIGO"].drop_duplicates().reset_index(drop = True).to_frame(name = "NUM")
    valid_dataset.to_csv("hall.csv")


def create_customer():
    filename = "T3H_TRANS_YSB_WW_DATA_TABLE.csv"
    dataset = pd.read_csv(filename)
    valid_dataset = dataset.loc[:,"CUST_ID"].drop_duplicates().reset_index(drop = True).to_frame(name = "CUST_ID")
    valid_dataset.to_csv("customer.csv")


        


if __name__ == '__main__':
    # create_customer()
    # create_hall()
    create_account()



