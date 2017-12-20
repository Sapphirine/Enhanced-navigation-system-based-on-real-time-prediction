Real time prediction based navigation system
1. Introduction
This framework implements traffic prediction into commonly used navigation systems. In that case, future traffic information will be further considered when performing the navigation.
- Applied pyspark MLlib to train the traffic models for prediction
- Performed secondary development on Dijkstar framework

2. Algorithm Procedures:
(1)Build Initial Original Street Graph according to NODE and EDGE lists.

(2)Get Shortest Nodes
- get the path from source to destination, containing smallest number of nodes (n).

(3)Get Subgraph according to source node and destination node.
- With parameter 'n_near', take any nodes whose sum of distance to source & destination nodes are within (n + n_near)

(4)Divide the Sub Graph into different time zones
- With parameter 'step_size', divide the whole subgraph. Beginning with source node, every 'step_size' of nodes will be classified into an individual region.

(5)Load Prediction Model and Features

(6)Update the weights of edges in each divided region with predictions of different times (1 hour traf, 2 hour traf...)

(7)Perform Dijkstra Algorithm for source and destination nodes on the updated subgraph.

3. Functions:
(1)Init_Graph(Node_list, Edge_list)
-Build a graph

Node_list: node's names in the graph [N1,N2,N3....]
Edge_list: edges in tuple [(N1, N2),(N2,N3)...]

Return a Graph

(2)Get_n_neighbors(Node, n, graph)
-Get the nodes and edges within n node-distance from center node

Node: center node
n: within n nearest neighbors
graph: any graph

Return a set and two lists
-record: a set containing each node's name, excluding center node
-SubNode: each node's name, including center node
-SubEdge: each Edge's name like (N1, N2)

(3)Get_n_m_neighbors(Node, n, m, graph)
-Get the nodes and edges from the difference set between n node-distance and m node-distance from center node

Node: center node
n: n nearest neighbors
m: m nearest neighbors
graph: any graph

Return two lists
-Node: each node's name
-Edge: each Edge's name like (N1, N2)

(4)Get_SubGraph(S_Node,D_Node,graph,n_near)
-Get a SubGraph from Original Graph according to minimize computations

S_Node: Source Node
D_Node: Destination Node
graph: Original Graph
n_near: How many more steps relative to shortest path are allowed

-Find the shortest node-path from source to destination
-extract all nodes and edges from original graph, whose distance sum to source and destination are within (n + n_near)

Return a Subgraph

(5)Region_Divider(S_Node, step_size, subgraph)
-Divide the subgraph into various regions according to step_size

S_Node: source node
step_size: node-distance relative to source node in each region 
subgraph: the subgraph

Return a list
-Region_List: list containing the Edge_List of each region.

(6)Predict_Update(model_list,subgraph,Region_List,edge_feature)
-Update edge weights in different regions using different prediction models that are trained for the corresponding time zone.
model_list: list of prediction models for different time zones
subgraph: the subgraph
Region_List: obtained from Region_Devider
edge_feature: a dictionary containing "key:value" of "edge_tuple:feature"

Return the updated Subgraph

(7) Navigation(S_Node,D_Node,graph)
-Perform navigation on the updated subgraph
S_Node: source node
D_Node: destination node
graph: the updated subgraph

Return Path_Info