from __future__ import print_function

from collections import defaultdict

import numpy as np
import random
import json
import sys
import os
import sklearn

import networkx as nx
from networkx.readwrite import json_graph

WALK_LEN = 5
N_WALKS = 50


def load_data(prefix, normalize=True, load_walks=True):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0

    for node in G.nodes():
        if not 'val' in G.nodes()[node] or not 'test' in G.nodes()[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():

        if (G.nodes()[edge[0]]['val'] or G.nodes()[edge[1]]['val'] or
                G.nodes()[edge[0]]['test'] or G.nodes()[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False



    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler

        train_ids = []
        val_ids = []
        test_ids = []
        for n in G.nodes():
            if not G.nodes()[n]['val'] and not G.nodes()[n]['test']:
                train_ids.append(n)
            elif G.nodes()[n]['val']:
                val_ids.append(n)
            else :
                test_ids.append(n)


        train_ids = np.array(train_ids)

        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:

            target = 0
            cum = []
            for line in fp:
                from_node, to_node = line.split()
                from_node, to_node = int(from_node), int(to_node)
                if from_node == target :
                    cum.append(to_node)
                else :
                    walks.append(cum)
                    target += 1
                    cum = []
                    cum.append(to_node)
            walks.append(cum)

    return G, feats, id_map, walks, class_map, train_ids, val_ids, test_ids

def get_neighbor_sample1(adj_list, num_sample):
    new_adj = []
    for i in range(len(adj_list)):
        if len(adj_list[i]) < num_sample:
            each = adj_list[i]
            each = np.array(each)
        else :
            each = np.random.choice(adj_list[i], num_sample, replace = False)
        new_adj.append(each.astype(np.int64))

    return new_adj


def get_neighbor_sample2(adj_list, N_u, num_sample):
    new_adj = []
    for i in range(len(adj_list)):

        candidate = []
        for j in N_u[i]:
            candidate = np.concatenate((candidate, N_u[j]))

        each = np.random.choice(candidate, num_sample, False)

        new_adj.append(each.astype(np.int64))

    return new_adj

def make_Beta_1(batch, N_u_1):
    new_list =  np.array(batch)
    for i in batch:
        new_list = np.concatenate((new_list, N_u_1[i]))

    return np.unique(new_list.astype(np.int64))

def make_Beta_0(batch, N_u_1, N_u_2):
    new_list = np.array(batch)
    for i in batch:
        new_list = np.concatenate((new_list, N_u_1[i], N_u_2[i]))

    return np.unique(new_list.astype(np.int64))


def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()

            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)


    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()

            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists



if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))