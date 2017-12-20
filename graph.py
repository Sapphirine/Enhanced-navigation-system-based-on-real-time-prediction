#################################################################
#       Real-Time Prediction Based Navigation System            #
# Project for "Big Data Analysis" Course at Columbia University #
#             Jim Yang, Sikai Zhou, Lei Shi                     #
#                                                               #
#################################################################

import csv
from dijkstar import Graph, find_path, single_source_shortest_paths, extract_shortest_path_from_predecessor_list
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf().setAppName("JimYang").setMaster("local").set("spark.driver.memory", "2G").set("spark.executor.memory", "1G")
sc = SparkContext(conf=conf)

cost_func = lambda u, v, e, prev_e: e['cost']

def Init_Graph(Node_list, Edge_list):
  graph_orig = Graph()
  
  # Build Graph with specified Nodes & Edges
  for nodes in Node_list:
    graph_orig.add_node(nodes)

  for edges in Edge_list:
    n1, n2 = edges
    graph_orig.add_edge(n1, n2, {'cost':1})
    graph_orig.add_edge(n2, n1, {'cost':1})

  return graph_orig


def Get_n_neighbors(Node, n, graph):
  if n == 0:
    return set()

  index = 0
  bool_1 = True
  bool_2 = False
  list_1 = [Node]
  list_2 = []
  record = set()

  while True:
    if index == n:
      Sub_Node = []
      
      if Node in record:
        record.remove(Node)

      for item in record:
        Sub_Node.append(item)
      Sub_Node.append(Node)

      # Sub_Node: contain Node
      # record: Node-free
      Sub_Node, Sub_Edge = Get_Edges(Sub_Node, graph)
      return record, Sub_Node, Sub_Edge 
    
    if bool_1:
      if len(list_1) == 0:
        bool_1 = False
        bool_2 = True
        index += 1
        continue
      
      Node_tmp = list_1[-1]
      list_1.pop()

      neighbors = graph.get(Node_tmp)
      for keys in neighbors:
        record.add(keys)
        list_2.append(keys)

    if bool_2:
      if len(list_2) == 0:
        bool_2 = False
        bool_1 = True
        index += 1
        continue
      
      Node_tmp = list_2[-1]
      list_2.pop()

      neighbors = graph.get(Node_tmp)
      for keys in neighbors:
        record.add(keys)
        list_1.append(keys)


def Get_n_m_neighbors(Node, n, m, graph):
  N,_,_ = Get_n_neighbors(Node, n, graph)
  M,_,_ = Get_n_neighbors(Node, m, graph)
  record = M - N

  _,Sub_Edge = Get_Edges(record, graph)
  
  for node in record:
    incoming_nodes = graph._incoming[node]
    for keys in incoming_nodes:
      if keys in N:
        Sub_Edge.append((keys, node))

  return record, Sub_Edge


def Get_Edges(node_set, graph):
  Sub_Node = []
  Sub_Edge = []
  tmp_set = set()
  for node in node_set:
    Sub_Node.append(node)
    neighbor = graph.get(node)
    for keys in neighbor:
      if keys in node_set:
        # Kill Repeated Edges
        if (node,keys) not in tmp_set:
          tmp_set.add((node, keys))
          tmp_set.add((keys, node))
          Sub_Edge.append((node, keys))
  return Sub_Node, Sub_Edge


def Get_Subgraph(S_Node, D_Node, graph, n_near):  
  # Calculate Number of Nodes in the shortest path
  path_info = find_path(graph, S_Node, D_Node, cost_func=cost_func)
  num_nodes = len(path_info.nodes) - 1
  record = set()

  # Obatain Set of Nodes within num_nodes + n_near
  num_select = num_nodes + n_near

  # Calculate for Distance_List
  S = single_source_shortest_paths(graph, S_Node, cost_func=cost_func)
  S_distance_list = {}
  for keys in S:
    distance = len(extract_shortest_path_from_predecessor_list(S, keys).nodes) - 1
    S_distance_list[keys] = distance

  D = single_source_shortest_paths(graph, D_Node, cost_func=cost_func)
  D_distance_list = {}
  for keys in D:
    distance = len(extract_shortest_path_from_predecessor_list(D, keys).nodes) - 1
    D_distance_list[keys] = distance
  
  # Calculate total distance from both S and D
  for node in S_distance_list:
    dis_s = S_distance_list[node]
    dis_d = D_distance_list[node]
    dis_s_d = dis_s + dis_d
    if dis_s_d <= num_select:
      record.add(node)

  record.add(S_Node)
  record.add(D_Node)
  
  # Build Subgraph based on record & Orig_Graph
  Sub_Node, Sub_Edge = Get_Edges(record, graph)
  sub_graph = Init_Graph(Sub_Node, Sub_Edge)

  return sub_graph


def Region_Divider(S_Node, step_size, subgraph):
  # Based on distance from S_Node, divide the whole graph into multiple regions
  Region_List = []
  _,_,Sub_Edge = Get_n_neighbors(S_Node,step_size,subgraph)
  
  Region_List.append(Sub_Edge)
  n = step_size
  while True:
    _,Sub_Edge = Get_n_m_neighbors(S_Node,n,n+step_size,subgraph)
    if len(Sub_Edge) == 0:
      return Region_List
    Region_List.append(Sub_Edge)
    n += 1


def Load_Features(path):
  edge_feature = {}
  with open(path, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
      Node1 = str(row[0])
      Node2 = str(row[1])
      features = []
      for i in range(2, len(row)):
        features.append(int(row[i]))
      edge_feature[(Node1,Node2)] = features
      edge_feature[(Node2,Node1)] = features

  return edge_feature


def Load_Regression_Model():
  model_1 = LogisticRegressionModel.load(sc, "./logistc_1.model")
  model_2 = LogisticRegressionModel.load(sc, "./logistc_2.model")
  model_3 = LogisticRegressionModel.load(sc, "./logistc_3.model")

  model_list = [model_1, model_2, model_3]
  return model_list


def Predict_Update(model_list, subgraph, Region_List, edge_feature):
  for i in range(len(Region_List)):
    # Use Prediction for all other Regions
    Edges = Region_List[i]
    for edge in Edges:
      if i == 0:
        feature = edge_feature[edge][0]
        cost = feature
      else:
        feature = edge_feature[edge][1:]
        if i - 1 >= 2:
          model = model_list[-1]
        else:
          model = model_list[i - 1]
        cost = model.predict(feature)
      
      subgraph.add_edge(edge[0],edge[1],{'cost':cost})
      subgraph.add_edge(edge[1],edge[0],{'cost':cost})
  return subgraph


def Navigation(S_Node, D_Node, subgraph):
  Path_Info = find_path(subgraph, S_Node, D_Node, cost_func=cost_func)
  return Path_Info


def Print_Path(path_info):
  nodes_list = path_info.nodes
  for node in nodes_list:
    print node
    if node != nodes_list[-1]:
      print '|'
      print '|'


def main():
  '''
  Example Graph:

  N1---N2
  |     |
  |     |
  |     |
  N3---N4---N5

  '''
  
  # Original Whole Graph Information (Nodes and Edges)
  NODE = ['N1','N2','N3','N4', 'N5']
  EDGE = [('N1','N2'),('N2','N4'),('N4','N3'),('N3','N1'),('N4','N5')]

  # Source and Destination Nodes for Navigation
  S_Node = 'N1'
  D_Node = 'N5'
  
  # Build Original Whole Graph
  orig_graph = Init_Graph(NODE,EDGE)

  # Extract a Subgraph for Fewer Computations
  subgraph = Get_Subgraph(S_Node, D_Node, orig_graph, 1)

  # Apply Region Devider
  Region_List = Region_Divider(S_Node, 1, subgraph)

  # Load Prediction Model
  Model_List = Load_Regression_Model()

  # Load Feature Data for Each Edge
  Edge_Features = Load_Features('./feature_example.csv')

  # Apply Prediction and Update Cost
  subgraph = Predict_Update(Model_List, subgraph, Region_List, Edge_Features)

  # Perform Navigation on Updated Subgraph
  path_info = Navigation(S_Node, D_Node, subgraph)
  
  Print_Path(path_info)
  
if __name__ == '__main__':
  main()
