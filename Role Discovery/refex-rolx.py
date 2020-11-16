"""
This module contains implementation of the paper on Rcursive feature extraction and role extraction -
https://web.eecs.umich.edu/~dkoutra/papers/12-kdd-recursiverole.pdf

Dataset - The collaboration network where nodes represent the authors and the
          the edge represents co-authorship.

Refex - Recursive Feature Extraction algorithm
1. Extract neighbourhood features (Degree and egonet fetures)
    - Degree - Number of neighbours of a node
    - Egonet features - 1. Egonet edges - Number of edges in the egonet of each node
                        2. Egonet Boundary edges - Number of edges at the boundary of egonet
                                                   (outside egonet , connected to nodes in the
                                                   egonet)
2. For each subsequent iteration , for each node perform aggregations
   sum and mean of each existing feature with neighbours and append them to existing
   node features
   Note: After k iterations we will have N*f^(k+1)
   where N is number of nodes and f is number of initial features captured for each node


After Refex, automatically identify the number of roles to extract using Minimum Description Length criterion.
Note: Did not implement this part, but from paper we know that there are 4 roles.

Rolx - Role extraction algorithm
1. From the given extracted feature matrix we need to compute the role membership
   of each node.
2. We use non-negative matrix factorization - GF ≈ V where each row of Gn×r represents
a node’s membership in each role and each column of Fr×f specifies how membership
in a specific role contributes to estimated feature values.
3. Based on the membership values of a node to a role we can assign a role to each node
   as the one with maximum membership.
"""

import json
import random
import snap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import networkx as nx
import itertools

default_aggs = [
    pd.DataFrame.sum,
    pd.DataFrame.mean,
]


def loadCollaborationGraph():
    """
    This method loads the collaboration graph
    return type : A TUNGraph
    :returns: Collaboration Graph
    """
    Graph = snap.TUNGraph.Load(snap.TFIn("collaboration.graph"))
    return Graph


def checkIsBoundary(edge, interiorNodes, exteriorNodes):
    """

    :param edge(tuple): tuple with srcNode, destNode
    :param interiorNodes(TIntV): Vector of internal node ids
    :param exteriorNodes(TIntV): Vector of exterior node ids
    :returns: True if edge is a boundary edge
    """
    v1, v2 = edge
    return (v1 in interiorNodes and v2 in exteriorNodes) or (v2 in interiorNodes and v1 in exteriorNodes)


def getEgonetFeatures(Graph, allNodeIds, nodeDeg, Node):
    """

    :param Graph (TUNGraph): A snap graph of Collaboration network
    :param allNodeIds(TIntV): Vector of all node ids
    :param nodeDeg(int): Degree of node under consideration
    :param Node(NodeI): Current node Iterator
    :returns: dictionary of egonet features
              - Number of egonet edges
              - Number of boundary edges outside egonet
    """
    interiorNodeIds = snap.TIntV()
    for i in range(nodeDeg):
        interiorNodeIds.Add(Node.GetNbrNId(i))

    if interiorNodeIds.Len() == 0:
        return {"egonet_edges": 0, "egonet_boundary_edges": 0}
    exteriorNodeIds = list(set(allNodeIds) - set(interiorNodeIds))
    SubGraph = snap.GetSubGraph(Graph, interiorNodeIds)
    edge_count = SubGraph.GetEdges()
    boundary_edges = 0
    for edge in Graph.Edges():
        if checkIsBoundary(edge.GetId(), interiorNodeIds, exteriorNodeIds):
            boundary_edges += 1
    return {"egonet_edges": edge_count, "egonet_boundary_edges": boundary_edges}


    """
    This is a better approach, found online
    tot_edges = 0
    nbrs = []
    for i in range(nodeDeg):
        nbrs.append(Graph.GetNI(Node.GetNbrNId(i)))
        tot_edges += nbrs[-1].GetDeg()
    inner_edges = 0
    for i in range(nodeDeg):
        for j in range(i):
            inner_edges += nbrs[i].IsInNId(nbrs[j].GetId())
    return {"egonet_edges": inner_edges, "egonet_boundary_edges": tot_edges - 2 * inner_edges}
    """

def extractNeighbourhoodFeatures(Graph):
    """
    This method returns a pandas dataframe of neighbourhood features (degree + egonet features)
    :param Graph: A TUNGraph object
    :returns: A pandas dataframe having neighbourhood features of each node in the graph.
    """
    node_feature_dict = dict()
    allNodeIds = snap.TIntV()
    for Node in Graph.Nodes():
        allNodeIds.Add(Node.GetId())

    for Node in Graph.Nodes():
        node = Node.GetId()
        node_feature_dict[node] = {}
        nodeDeg = Node.GetDeg()
        node_feature_dict[node]["degree"] = nodeDeg
        node_feature_dict[node].update(getEgonetFeatures(Graph, allNodeIds, nodeDeg, Node))
    return pd.DataFrame.from_dict(node_feature_dict, orient='index')


def updateNodeFeatures(new_features, existing_features):
    """
    Update existing node features with new features from aggregations.
    :param new_features(pd.DataFrame): new features of nodes after each iteration (sum and average aggregations)
    :param existing_features(pd.DataFrame): existing features of nodes, initially empty
    :returns: A pandas dataframe concatenating old features and newly computed features
    """
    return (pd.concat([existing_features, new_features], axis=1, sort=True)
            .fillna(0))


def aggreatedDfToDict(agg_df, iter_num):
    """
    The pandas dataframe containing previous iteration features along with sum and mean aggregations
    of the existing features
    :param agg_df(pd.DataFrame): Dataframe containing sum and mean of existing features
    :param iter_num(int): Iteration number
    :returns:
    """
    try:
        agg_dicts = agg_df.to_dict(orient='index')
    except TypeError:
        # pd.Series objects do not have to_dict method with orient kwarg
        # so cast to pd.DataFrame and transpose for correct shape
        agg_dicts = agg_df.to_frame().T.to_dict(orient='index')
    formatted_agg_dict = {
        f'{key}({idx})({iter_num})': val
        for idx, row in agg_dicts.items()
        for key, val in row.items()
    }
    return formatted_agg_dict


def extractRecursiveFeatures(Graph, num_iterations=2):
    """
    Returns pandas dataframe containing the final features after k iterations
    For k iterations,  pandas dataframe shape - (N, 3^(k+1))
    :param Graph: A TUNGraph
    :param num_iterations: Number of iterations
    :returns: A pandas data frame containing final features for each node.
    """
    final_features = pd.DataFrame()
    feature_iter = {}
    neigh_features = extractNeighbourhoodFeatures(Graph)
    final_features = updateNodeFeatures(neigh_features, final_features)
    feature_iter[0] = final_features.to_dict()
    for i in range(1, num_iterations + 1):
        prev_features = feature_iter[i - 1].keys()
        new_features = {
            node.GetId(): (
                final_features
                    .reindex(index=[node.GetNbrNId(j) for j in range(node.GetDeg())], columns=prev_features)
                    .agg(default_aggs)
                    .fillna(0)
                    .pipe(aggreatedDfToDict, i)
            )
            for node in Graph.Nodes()
        }
        new_features = pd.DataFrame.from_dict(new_features, orient='index')
        final_features = updateNodeFeatures(new_features, final_features)
        feature_iter[i] = final_features.to_dict()
    return final_features


def getTopKSimilarNodes(simMatrix, nodeId=9, k=5):
    """
    Get top k similar nodes for a given node based on cosine similarity
    :param simMatrix (list of lists): N * N cosine similarity matrix
    :param nodeId9(int): Id of node under consideration
    :param k(int): number of similar nodes to find
    :return sim_k(pd.DataFrame): Dataframe containing k sim nodes
    """
    cos_sim_df = pd.DataFrame(simMatrix)
    cos_sim_k = pd.DataFrame({n: cos_sim_df.T[column].nlargest(k + 1).index.tolist()
                              for n, column in enumerate(cos_sim_df.T)}).T
    for i in range(k + 1):
        print(cos_sim_df.iloc[nodeId, cos_sim_k.iloc[nodeId, i]])
    return cos_sim_k.iloc[[nodeId]]


def getD3JsonFromGraphNMF(Graph, node_role_matrix):
    """
    Assign role based on the membership value of node to a role (use maximum)
    """
    node_role_map = node_role_matrix.argmax(axis=1).tolist()
    graph_json = {'directed': False, 'links': [], 'nodes': []}
    for edge in Graph.Edges():
        src, dest = edge.GetId()
        graph_json['links'].append({"source": src, "target": dest})

    for Node in Graph.Nodes():
        node_id = Node.GetId()
        graph_json['nodes'].append({"id": node_id, "group": node_role_map[node_id]})

    with open("graph_NMF.json", "w") as fp:
        json.dump(graph_json, fp=fp)


def NMFallFeatures(allFeatures):
    """
    Non-negative matrix factorization with number of components
    as 4 (number of roles)
    """
    np_features = allFeatures.to_numpy()
    model = NMF(n_components=4, solver='mu', init='nndsvda')
    W = model.fit_transform(np_features)
    return np.array(W)


Graph = loadCollaborationGraph()
allFeatures = extractRecursiveFeatures(Graph)
node_role_matrix = NMFallFeatures(allFeatures)
getD3JsonFromGraphNMF(Graph, node_role_matrix)

"""
Homework part below -

There are mainly 4 roles - 
 Members of Clique - nodes of these role are mostly part of dense / fully connected network
   - In the context of co-authorship , think of top papers that were published by a group of 
     senior professors collaborating together (maybe few students are involved too, they are peripherals) 
 Centers of stars - nodes that connect groups of nodes
   - In the context of co-authorship, maybe faculty, advisors or researchers.
 Peripherals (mainstream) - Nodes that are at edge of the graphs and are having few connections
   - In the context of co-authorship, it may students who during a short tenure( 2-5 yrs) would
     collaborate with some of the faculty in the university.
 Extreme peripherals - nodes that have very less degree and are at extreme edges of the graph
   - In the context of co-authorship, these are isolated authors with very few connections.

    In histogram consider node 9 - 
    4 roles -  1. nodes with similarity ~ 0.40 and 0.45 (clique)
               2. nodes with similarity ~ 0.60 and 0.65 (pathy)
               3. nodes with similarity ~ 0.85 and 0.90 (centre of stars)
               4. nodes with similarity ~ 0.9 and 1.0 (mainstream, node 9 belongs here)
   Now construct hypothetical network structures using networkx for each role node by picking a node in each role.  
   Construct the structure from it's neighbourhood features.
"""


def getD3JsonFromGraph(Graph, role_nodes):
    """
    We see 4 spikes in the histogram of node similarity of nodes wrt node 9.
    Based on these four ranges we assign nodes different roles and create d3 graph.
    """
    graph_json = {}
    graph_json['directed'] = False
    graph_json['links'] = []
    graph_json['nodes'] = []
    role_dict = {}
    for i, nodes in enumerate(role_nodes):
        for node in nodes:
            role_dict[node] = i
    for edge in Graph.Edges():
        src, dest = edge.GetId()
        graph_json['links'].append({"source": src, "target": dest})

    for Node in Graph.Nodes():
        node_id = Node.GetId()
        if node_id not in role_dict:
            graph_json['nodes'].append({"id": node_id, "group": 4})
            continue
        graph_json['nodes'].append({"id": node_id, "group": role_dict[node_id]})

    with open("graph_hist.json", "w") as fp:
        json.dump(graph_json, fp=fp)


def getNodesFromEachRole(simFeatureMatrix):
    """
    Based on the similarity scores wrt node 9, get nodes from different roles.
    """
    arr_sim = np.array(simFeatureMatrix[9][:])
    role1 = 0.40, 0.45  # clique
    role2 = 0.60, 0.65  # pathy
    role3 = 0.75, 0.90  # centre of stars
    role4 = 0.90, 0.95  # mainstream

    role1_nodes = np.where(np.logical_and(arr_sim >= role1[0], arr_sim < role1[1]))[0].tolist()
    role2_nodes = np.where(np.logical_and(arr_sim >= role2[0], arr_sim < role2[1]))[0].tolist()
    role3_nodes = np.where(np.logical_and(arr_sim >= role3[0], arr_sim < role3[1]))[0].tolist()
    role4_nodes = np.where(np.logical_and(arr_sim >= role4[0], arr_sim < role4[1]))[0].tolist()

    return [role1_nodes, role2_nodes, role3_nodes, role4_nodes]


def genGraphsForRoleNode(Node, role, features):
    """
    Pick a node from each role and draw a graph of the node and it's neighbourhood
    """
    nodeId = Node.GetId()
    degree = features.loc[nodeId, 'degree']
    egonet_edges = features.loc[nodeId, 'egonet_edges']
    egonet_boundary_edges = features.loc[nodeId, 'egonet_boundary_edges']

    g = nx.Graph()
    main_node = f'main-{nodeId}'
    g.add_node(main_node)

    neigh1_nodes = []
    for i in range(degree):
        neigh_node = f'neigh-1-{i}'
        neigh1_nodes.append(neigh_node)
        g.add_node(neigh_node)
        g.add_edge(main_node, neigh_node)

    while egonet_edges > 0:
        src = random.choice(neigh1_nodes)
        dest = random.choice(neigh1_nodes)
        if src != dest and (not g.has_edge(src, dest) or not g.has_edge(dest, src)):
            g.add_edge(src, dest)
            egonet_edges -= 1

    neigh2_nodes = []
    for i in range(0, egonet_boundary_edges):
        neigh_node = f'neigh-2-{i}'
        neigh2_nodes.append(neigh_node)
        g.add_node(neigh_node)

    while egonet_boundary_edges > 0:
        src = random.choice(neigh1_nodes)
        dest = random.choice(neigh2_nodes)
        if src != dest and (not g.has_edge(src, dest) or not g.has_edge(dest, src)):
            g.add_edge(src, dest)
            egonet_boundary_edges -= 1

    deg = g.degree()
    to_remove = [n for n, deg_val in deg if deg_val == 0]
    g.remove_nodes_from(to_remove)
    colors = []
    for node in g.nodes():
        if node == main_node:
            colors.append('r')
        else:
            colors.append('b')
    nx.draw(g, node_color=colors)
    plt.savefig(f'graph-{role}.png', bbox_inches='tight')
    plt.close()


cols = ['degree', 'egonet_edges', 'egonet_boundary_edges']
neighFeatures = extractNeighbourhoodFeatures(Graph)
print(f"Features of node 9 : {allFeatures.loc[[9], cols]}")
simNeighFeatures = cosine_similarity(neighFeatures)
print("Top 5 nodes similar to node 9")
sim_k1 = getTopKSimilarNodes(simMatrix=simNeighFeatures, nodeId=9, k=5)

simAllFeatures = cosine_similarity(allFeatures)
X = simAllFeatures[9][:]
plt.hist(X, 20, histtype='bar')
plt.xlabel('Cosine similarities for node 9')
plt.ylabel('Number of Nodes')
plt.title('Histogram of number of nodes vs cosine similarities for node 9')
plt.savefig('node_sim_histogram.png')
plt.close()

role_nodes = getNodesFromEachRole(simAllFeatures)
getD3JsonFromGraph(Graph, role_nodes)
role1_node, role2_node, role3_node, role4_node = random.choice(role_nodes[0]), random.choice(role_nodes[1]), \
                                                 random.choice(role_nodes[2]), random.choice(role_nodes[3])

print(f"Feature vector of node  {role1_node} : {allFeatures.loc[[role1_node], cols]}")
genGraphsForRoleNode(Graph.GetNI(role1_node), f'{role1_node}', allFeatures.loc[[role1_node], cols])
print(f"Feature vector of node  {role2_node} : {allFeatures.loc[[role2_node], cols]}")
genGraphsForRoleNode(Graph.GetNI(role2_node), f'{role2_node}', allFeatures.loc[[role2_node], cols])
print(f"Feature vector of node  {role3_node} : {allFeatures.loc[[role3_node], cols]}")
genGraphsForRoleNode(Graph.GetNI(role3_node), f'{role3_node}', allFeatures.loc[[role3_node], cols])
print(f"Feature vector of node  {role4_node} : {allFeatures.loc[[role4_node], cols]}")
genGraphsForRoleNode(Graph.GetNI(role4_node), f'{role4_node}', allFeatures.loc[[role4_node], cols])
print(f"Feature vector of node {9} : {allFeatures.loc[[9], cols]}")
genGraphsForRoleNode(Graph.GetNI(9), 'node-9', allFeatures.loc[[9], cols])
