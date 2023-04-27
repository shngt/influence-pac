#!/usr/bin/env python
# coding: utf-8


from joblib import dump, load
import numpy as np
import math
import random
import snap
import cvxpy as cp
import multiprocessing as mp
from math import pow
import networkx as nx
import itertools
import datetime


def genR_EdgeWeights(G):
    probabilities = {}
    for edge in G.Edges():
        u = edge.GetSrcNId()
        v = edge.GetDstNId()
        weight = random.random()
        probabilities[(u, v)] = weight
    return probabilities

def sample_power_law(U):
    
    K = int(np.random.power(0.6) * len(U))
    R = random.sample(U, K)
    
    return set(R)

def independent_cascade(G, p, S):
    """
    Runs the independent cascade model on a directed graph G with edge probabilities p
    starting from a seed set S. Returns the set of activated nodes O.

    Args:
    - G: a directed graph (snap.TNGraph)
    - p: a dictionary of edge weights in [0,1] (keys are (source, target) tuples, values are floats)
    - S: a set of node IDs representing the initial seed set

    Returns:
    - O: a set of node IDs representing all the activated nodes throughout the algorithm
    """
    # Initialize the sets
    A = S.copy() # set of active nodes
    B = set()    # set of newly activated nodes
    O = S.copy()   # set of all activated nodes
    
    count=0
    while count < G.GetNodes() and A:
        # Iterate over the active nodes
        for u in A:
            # Iterate over u's neighbors
            for v in G.GetNI(u).GetOutEdges():
                # Check if v has already been activated
                if v in O:
                    continue
                # Check if the edge (u,v) is activated
                edge_prob = p.get((u, v), 0)
                if random.random() < edge_prob:
                    B.add(v)
                    O.add(v)
        # Update the set of active nodes
        A = B.copy()
        B.clear()
        count+=1

    return O


def compute_y(num_nodes, realizations):
    
    y = {} # dictionary which each entry y_ij = 1 if node j is in B_i for (A_i,B_i), and 0 otherwise
    
    
    for j in range(num_nodes): # Fix a node
        i = 0
        for (A,B) in realizations: # Iterate through each sample, check if node is in it, tracking samples using index i
            #if j in realizations[i][1]:
            if j in B:
                y[(i,j)] = 1
            i += 1

    return y


# In[6]:


def compute_q(num_nodes, realizations, y):
    
    q = {} # dictionary which each entry q_ij is the frequency that node j was observed as active over all times node i was in seed
    
    # Fix a node a graph
    for i in range(num_nodes):
        
        # We find all realizations that are seeded with i, aggregating their indices to a set
        seeds_with_i = set()
        realiz_idx = 0
        for (A,B) in realizations:
            if i in A: seeds_with_i.add(realiz_idx)
            realiz_idx += 1
        
        # if set empty we skip
        if not seeds_with_i: continue
            
        # now, for each node j, we find the frequency of i being seed, and j being infected
        for j in range(num_nodes):
            
            # skip case j == i
            if j == i: continue
            
            count_j = 0
            # check and add if node j is in A_i
            for idx in seeds_with_i:
                count_j += y.get((idx,j),0)
            
            # the fraction of times j was infected when i was in the seed
            q[(i,j)] = count_j/len(seeds_with_i)

    return q


def kitchen_sink(j,q,K,d):
    """
    Generates a list of random binary feature vectors (See Sec 4. of Du)
    
    Args:
    -j: node of graph under consideration
    -q: dictionary which each entry q_ij is the fraction of times that "node j in actived set" occurs over all samples where "node i was in seed"
    -K: InfluLearner parameter, number of random vectors
    -d: dimension of graph, i.e. number of nodes in V
    
    Returns:
    - features: list of random vectors in {0,1}^d
    
    """
    features = [] # initialization
    
    for i in range(K):
        
        v = np.zeros(d)
        for u in range(d):
            
            # for each entry, with prob. q_uj make entry '1'
            prob_u = q.get((u,j),0)
            
            if random.random() < prob_u:
                v[u] = 1
                
        # add vector to list, iterate
        features.append(v)

    return features


# In[8]:


def true_influence_benchmark(G,p,numSeeds, averaging_repititions):
    
    test_seeds = []
    for _ in range(numSeeds):
        
        seed = sample_power_law(set(range(G.GetNodes())))
        avg = 0
        for _ in range(averaging_repititions):
            avg += len(independent_cascade(G,p,seed))
            
        test_seeds.append((seed,avg/averaging_repititions)) 

    return test_seeds


def InfluLearnerMaster(q,y,train,K,lambd,d):
    
    """
    Implements InfluLearner algorithm using CVXPY for exponentiated gradient optimization subroutine.
    
    Args:
    -X: seed of interest.
    -q: q_ij's built through compute_q.
    -y: y_ij's built through compute_y.
    -K: number of kitchen_sink functions. K = 100 in original implementation.
    -lambd: lambda for winsorized functions (makes sure)
    -d: graph dimension

    Returns:
    - sum(results): influence as computed by influLearner
    
    """
    
    
    #Assign thread jobs (instead of iterating over each j in [d])
    num_processes = mp.cpu_count()
    sub_tasks = np.array_split(range(d), num_processes)
    sub_lists = [sub_list.tolist() for sub_list in np.array_split(range(d), num_processes)]
    
    # Same method as in trueInfluMaster
    # Each thread assignment calls the same random number generator seeded at 42, so to avoid them
    # being coupled (i.e. using same random bits), we a priori create a list of kitchen_sink functions
    # where all_sinks[i] contains a list of the kitchen sink functions to be used for node i
    # This ensures reproducibility
    
    all_sinks = []
    for i in range(d):
        all_sinks.append(kitchen_sink(i,q,K,d))
    
    # Assign threads, giving each all_sinks so they can find the correct kitchen_sinks function list for their sub-task nodes.
    with mp.Pool(processes=num_processes) as pool:
        results = list(pool.map(InfluLearnerWorker, [(sub_lists[i],all_sinks,q,y,train,K,lambd,d) for i in range(num_processes)]))
    
    w_vec = {}
    for dictionary in results:
        w_vec.update(dictionary)
    
    return w_vec,all_sinks



def Du_sampleset_Master(G,p,numSeeds,numCascadesPerSeed):
    
    # convert graph to networknx graph, which can be pickled
    G_nx = nx.DiGraph()
    for node in G.Nodes():
        G_nx.add_node(node.GetId())
    for edge in G.Edges():
        G_nx.add_edge(edge.GetSrcNId(), edge.GetDstNId())
    

   #initialize all seeds
    seeds = []
    for _ in range(numSeeds):
        seeds.append(sample_power_law(set(range(G.GetNodes()))))
        
    print("seeds generated...")
    

    num_processes = mp.cpu_count()
    sublists = np.array_split(seeds, num_processes)
    sublists_of_sets = [list(subarray) for subarray in sublists]

    with mp.Pool(processes=num_processes) as pool:
        results = list(pool.map(Du_sampleset_Worker, [(sublists_of_sets[i], G_nx, p, numCascadesPerSeed) for i in range(num_processes)]))
    
    print("flattening result...")
    flattened_results = list(itertools.chain.from_iterable(results))
    return flattened_results


def tester(G,p, sample_sizes, test_set):
    
    out = {}
    
    for sample_size in sample_sizes:
        
        print("started generating set for " + str(sample_size) + "samples")
        train_set = Du_sampleset_Master(G,p,sample_size,150) # get 100 cascade per seed
        print("obtained training set for " + str(sample_size) + "samples")
        
        y = compute_y(G_er.GetNodes(),train_set)
        q = compute_q(G_er.GetNodes(),train_set, y)
        print("starting influlearner training...")
        w_vec,all_sinks = InfluLearnerMaster(q,y,train_set,150,0.05,G.GetNodes())
        print("finished influlearner training...")
        
        print("beginning mae compute")
        mae = 0
        for idx in range(len(test_set)):
            
            est = influModelCompute(test_set[idx][0],w_vec,all_sinks,G,150,0.05)
            real = test_set[idx][1]
            mae += abs(est - real)
        
        out[sample_size] = mae/len(test_set)
        print("done with mae compute")
    
    return out
        
def main():

    from trueInfluence_v3 import compute_true_infl
    from influFunctions_v3 import InfluLearnerWorker,influModelCompute,Du_sampleset_Worker

    G_er = snap.GenRndGnm(snap.TNGraph, 100,250, True)
    p_er = genR_EdgeWeights(G_er)

    test_benchmark = true_influence_benchmark(G_er,p_er,100,1000)

    start_time = datetime.datetime.now()

    # mae = tester(G_er, p_er, [1, 2, 5, 10, 25, 75, 100, 200, 500, 1000], test_benchmark)
    mae = tester(G_er, p_er, [1, 2, 5, 10, 25], test_benchmark)

    end_time = datetime.datetime.now()
    runtime = end_time - start_time
    print("Total runtime:", runtime)

if __name__ == '__main__':
    main()



