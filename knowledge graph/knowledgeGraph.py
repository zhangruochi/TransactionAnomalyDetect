from neo4j import GraphDatabase
import pickle

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "lv23623600"))

"""
def add_friend(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
           "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
           name=name, friend_name=friend_name)

def print_friends(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                         "RETURN friend.name ORDER BY friend.name", name=name):
        print(record["friend.name"])

"""


def add_hall(tx):
    """添加营业厅
    """
    tx.run("LOAD CSV WITH HEADERS  FROM \"file:///hall.csv\" AS line MERGE (hall:Hall{NUM:line.NUM})")


def add_teller(tx):
    """添加柜员
    """
    tx.run("LOAD CSV WITH HEADERS  FROM \"file:///F_CM_KTLP_GYCSHU_DATA_TABLE.csv\" AS line MERGE (teller:Teller{GUIYDAIH:line.GUIYDAIH,hall:line.YNGYJIGO,attribute:line.GUIYSHUX,department:line.BUMENHAO,type:line.GUIYLXIN,class:line.GUIYJBIE})")


def add_account(tx):
    """添加账户
    """
    tx.run("LOAD CSV WITH HEADERS  FROM \"file:///account.csv\" AS line MERGE (account:Account{ACCT_ID:line.ACCT_ID})")


def add_customer(tx):
    """添加客户
    """
    tx.run("LOAD CSV WITH HEADERS  FROM \"file:///customer.csv\" AS line MERGE (customer:Customer{CUST_ID:line.CUST_ID})")


def add_operation_relation(tx):
    """添加柜员与账号之间的操作关系
    """
    tx.run("LOAD CSV WITH HEADERS  FROM \"file:///T3H_TRANS_YSB_WW_DATA_TABLE.csv\" AS line match (from:Teller{GUIYDAIH:line.OPR_ID}),(to:Account{ACCT_ID:line.ACCT_ID}) merge (from)-[r:OPERATION]->(to)")


def add_transaction_relation(tx):
    """添加账号与账号之间的交易关系
    """
    tx.run("LOAD CSV WITH HEADERS  FROM \"file:///T3H_TRANS_YSB_WW_DATA_TABLE.csv\" AS line match (from:Account{ACCT_ID:line.ACCT_ID}),(to:Account{ACCT_ID:line.OPP_ACCT_ID}) merge (from)-[r:transaction{time:line.TR_TM,number:line.TR_NO,customer:line.CUST_ID,amount:line.TR_AMT,balance:line.TR_BAL_AMT,operationID:line.OPR_ID,reOperationID:line.RE_OPR_ID,customerType:line.CUST_TYPE}]->(to)")


def add_have_rela(tx):
    """添加账号与客户之间的拥有关系(一个account拥有多个customerID)
    """
    tx.run("LOAD CSV WITH HEADERS  FROM \"file:///T3H_TRANS_YSB_WW_DATA_TABLE.csv\" AS line match (from:Account{ACCT_ID:line.ACCT_ID}),(to:Customer{CUST_ID:line.CUST_ID}) merge (from)-[r:Have]->(to)")



def add_supervised_result(tx):
    """ add supervised learning result
    """
    with open("supervised_result.pkl","rb") as f:
        result_dict = pickle.load(f)
    for account,score in result_dict.items():
        tx.run("MATCH (acc:Account) where acc.ACCT_ID = '{}' set acc.supervised_socre = {} return acc;".format(account,score))


    with open("predicted_label.pkl","rb") as f:
        label_dict = pickle.load(f)
    for account,label in label_dict.items():
        tx.run("MATCH (acc:Account) where acc.ACCT_ID = '{}' set acc.supervised_label = {} return acc;".format(account,label))


def add_unsupervised_result(result_dict,tx):
    """ add unsupervised learning result
    """
    with open("unsupervised_result.pkl","r") as f:

        result_dict = pickle.load(f)
    for trans,score in result_dict.items():
        tx.run("MATCH p=()-[r:transaction]->() SET r.abnormal_score = {} where r.TR_NO = {} RETURN p LIMIT 25".format(score,trans))


with driver.session() as session:
    #session.write_transaction(add_friend, "Arthur", "Guinevere")

    # session.write_transaction(add_hall)
    # session.write_transaction(add_teller)
    # session.write_transaction(add_account)
    # session.write_transaction(add_customer)
    # session.write_transaction(add_have_rela)
    # session.write_transaction(add_operation_relation)
    # session.write_transaction(add_transaction_relation)

    # session.write_transaction(add_unsupervised_result)
    session.write_transaction(add_supervised_result)





    