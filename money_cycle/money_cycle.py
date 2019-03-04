from neo4j import GraphDatabase
import pickle
import networkx
from neo4j import GraphDatabase
import pickle


def read_nodes(tx):
    res = tx.run(
        """
        MATCH (n:Account)-[r1:transaction]->(m:Account) 
        WITH count(n) as InDgree,n  
        match(m)-[r:transaction]->(p:Account)
        with r,InDgree,count(p) as OutDgree,m 
        where InDgree >= 1 and OutDgree >= 1
        return m     
        """)
    return res
   

def read_graph(tx):
    res = tx.run("MATCH (n:Account)-[r:transaction]-() RETURN n, r")  
    return res


def rs2graph(G,all_nodes):
    graph = networkx.MultiDiGraph()


    for record in G:
        node = record['n']
        if node and node.id in all_nodes :
            #print("adding node")
            nx_properties = {}
            nx_properties.update(dict(node))
            nx_properties['labels'] = node.labels
            graph.add_node(node.id, **nx_properties)

        relationship = record['r']
        if relationship is not None:   # essential because relationships use hash val
            #print("adding edge")
            graph.add_edge(
                relationship.start_node.id, relationship.end_node.id, key=relationship.type
            )
    return graph


if __name__ == '__main__':
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "lv23623600"))
    n = 4

    with driver.session() as neo:
        nodes = neo.write_transaction(read_nodes)
        G = neo.write_transaction(read_graph)
    
    all_nodes = set()
    for node in nodes:
        all_nodes.add(node["m"].id)

    graph = rs2graph(G,all_nodes)

    print(graph.number_of_nodes())
    print(graph.number_of_edges())
    
    try:
        circles = networkx.simple_cycles(graph)
    except:
        print("no circle")

    for cirle in circles:
        if len(cirle) >= n:
            print(cirle)  


    with open("circles.pkl","wb") as f:
        pickle.dump(list(circles),f)