#################################################################
#       Real-Time Prediction Based Navigation System            #
# Project for "Big Data Analysis" Course at Columbia University #
#             Jim Yang, Sikai Zhou, Lei Shi                     #
#                                                               #
#################################################################

import csv
import osmnx as ox
import matplotlib.pyplot as plt

from threading import Thread
from multiprocessing import Queue
from time import sleep
from dijkstar import Graph, find_path, single_source_shortest_paths, extract_shortest_path_from_predecessor_list
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf

q = Queue()
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

  return sub_graph, Sub_Edge


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
    n += step_size


def Load_Features(path):
  edge_feature = {}
  with open(path, 'rt', encoding='ascii') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
      Node1 = int(row[0])
      Node2 = int(row[1])
      
      features = row[2:]
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
  cost_dict = {}
  print ('Prediction Start!')
  for i in range(len(Region_List)):
    sleep(1)
    print ('--------------------------------------')
    Edges = Region_List[i]
    if i == 0:
      print ('Use Current Traffic Level for Region: ', i)
    else:
        print ('Performing Prediction for Region: ', i)
        print ('Load Model Number: ', i)
    for edge in Edges:
      feature = edge_feature[edge]
      if i == 0:
        if int(feature[1]) >= 1000:
          cost = 10
        elif int(feature[1]) >= 100:
          cost = 5
        else:
          cost =1
          
        print ('Street Number: ', edge)
        print ('Apply Traffic Level: ', cost)

        cost_dict[(edge[0],edge[1])] = cost
        cost_dict[(edge[1],edge[0])] = cost

        subgraph.add_edge(edge[0],edge[1],{'cost':cost})
        subgraph.add_edge(edge[1],edge[0],{'cost':cost})
      
      else:
        if i >= 3:
          model = model_list[-1]
        else:
          model = model_list[i]
        
        level = model.predict(feature)
        if level == 2:
          cost = 10
        elif level == 1:
          cost = 5
        elif level == 0:
          cost = 1

        print ('Street Number: ', edge)
        print ('Predicted Traffic Level: ', cost)

        cost_dict[(edge[0],edge[1])] = cost
        cost_dict[(edge[1],edge[0])] = cost

        subgraph.add_edge(edge[0],edge[1],{'cost':cost})
        subgraph.add_edge(edge[1],edge[0],{'cost':cost})
  return subgraph, cost_dict


def Navigation(S_Node, D_Node, subgraph):
  Path_Info = find_path(subgraph, S_Node, D_Node, cost_func=cost_func)
  return Path_Info


def Print_Path(path_info):
  nodes_list = path_info.nodes
  for node in nodes_list:
    print (node)
    if node != nodes_list[-1]:
      print ('|')
      print ('|')

def insert(q, plt, NODE): 
  while (True):
    sleep(1)
    S_Node = int(input("source_node: "))
    D_Node = int(input("destination_node: "))
    if S_Node in NODE and D_Node in NODE:
      print ("Insert Accepted!")
      q.put([S_Node,D_Node])
      plt.close()
      break
    else:
      print ("Non-Existing Nodes, Please Insert Again")

def press_continue(plt):
  input("Proceed To Next Step")
  plt.close()

def main():
  '''
  Example Graph:

  N1---N2
  |     |
  |     |
  |     |
  N3---N4---N5

  '''
  # Build Graph in OSMNX
  location_point = (40.683646,-73.946851)
  G = ox.graph_from_point(location_point, timeout=180, distance=700, distance_type='network', network_type='walk')
  
  # Original Whole Graph Information (Nodes and Edges)
  NODE = []
  EDGE = []
  for nodes in G.nodes():
    NODE.append(int(nodes))
  for edges in G.edges():
    edge_tmp = (int(edges[0]),int(edges[1]))
    EDGE.append(edge_tmp)

  # Source and Destination Nodes for Navigation
  # Input Part
  
  thread1 = Thread(target=insert,args=(q,plt,NODE))
  thread1.start() 
  ox.plot_graph(G, fig_height=50, fig_width=50,annotate=True, edge_color='black')
  thread1.join()

  S_Node, D_Node = q.get()
  q.close()

  # Build Original Whole Graph
  orig_graph = Init_Graph(NODE,EDGE)

  # Extract a Subgraph for Fewer Computations
  subgraph,subedge = Get_Subgraph(S_Node, D_Node, orig_graph, 2)

  #thread2 = Thread(target=press_continue,args=(plt,))
  #thread2.start()
  #ox.plot.plot_graph_color(G, [subedge], [S_Node, D_Node], fig_height=50, fig_width=50, edge_color = 'grey', annotate=True)
  #thread2.join()

  # Apply Region Devider
  Region_List = Region_Divider(S_Node, 3, subgraph)
  
  #thread3 = Thread(target=press_continue,args=(plt,))
  #thread3.start()
  #ox.plot.plot_graph_color(G, Region_List, [S_Node, D_Node], fig_height=50, fig_width=50, edge_color = 'grey', annotate=True)
  #thread3.join()

  # Load Prediction Model
  Model_List = Load_Regression_Model()

  # Load Feature Data for Each Edge
  Edge_Features = Load_Features('./feature_example.csv')

  # Apply Prediction and Update Cost
  subgraph,cost_dict = Predict_Update(Model_List, subgraph, Region_List, Edge_Features)
  
  update_list = []
  for i in range(len(Region_List)):
    for edges in Region_List[i]:
      update_list.append(edges)
    
    #thread = Thread(target=press_continue,args=(plt,))
    #thread.start()
    #ox.plot.plot_graph_update(G, [update_list], [S_Node, D_Node], cost_dict, fig_height=50, fig_width=50, edge_color = 'grey', annotate=True)
    #thread.join()

  # Perform Navigation on Updated Subgraph
  path_info = Navigation(S_Node, D_Node, subgraph)
  route = path_info.nodes
  
  ox.plot.plot_graph_route(G, route, [update_list], [S_Node, D_Node], cost_dict, route_color='navy', fig_height=50, fig_width=50, edge_color = 'grey', annotate=True)
  
if __name__ == '__main__':
  main()
