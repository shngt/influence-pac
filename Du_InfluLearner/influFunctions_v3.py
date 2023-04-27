import cvxpy as cp
import numpy as np
import snap
import networkx
import random

def incidence_conversion(S, d):
    """
    Convert a set S to a d-dimensional incidence vector
    
    Parameters:
    S (set): input set
    d (int): size of the universal set
    
    Returns:
    numpy.ndarray: incidence vector representing S
    """
    vec = np.zeros(d, dtype=int)
    for i in S:
        vec[i] = 1
    return vec

def phi(a):
    """
    Returns minimum of argument and 1.
    
    """
    return min(a,1)

def Phi_vect(inc_S,r,K):
    
    """
    Performs "line 4" of InfluLearner for the seed of the ith training example.
    
    Called Functions:
    -phi()
    
    Args:
    -inc_S: incidence vector of a set S with respect to some universal set.
    -r: K-length list of kitchen_sink functions
    -K: InfluLearner parameter
    
    Returns:
    - v: Vector where the ith entry is result of applying phi() to <inc_S,r_i>
    
    """
    v = np.zeros(K) # initialize output
    for i in range(K):
        inner_prod = np.dot(inc_S,r[i])
        v[i] = phi(inner_prod) # apply phi()
    return v

def InfluLearnerWorker(args):
    
    # Parse args passed by mp
    sub_list,all_sinks,q,y,train,K,lambd,d = args
    
    m = len(train)
    
#     X_inc = incidence_conversion(X,d) # converts X to incidence vector
    
    dict_w = {}
    
    # For each j in sub tasks, perform outer for loop of Du et al.
    for j in sub_list:

        train_Phi = [] # list of all of "line 4" vectors from training samples
        r = all_sinks[j] # sample K random feature vectors from distribution q
#         X_Phi = Phi_vect(X_inc,r,K) # ith component from projecting X_inc onto each r_i and applying phi(.)

        # line 4, populating train_Phi 
        for i in range(m):
            Si = train[i][0]
            Si_inc = incidence_conversion(Si,d)
            train_Phi.append(Phi_vect(Si_inc,r,K))

        # Define the optimization variables, initialize w_1 to interior point of K-simplex
        w = cp.Variable(K, value=np.ones(K)/K)

        # Define the objective function
        obj = 0
        for i in range(len(train_Phi)):
            p = (1-2*lambd)*cp.matmul(w, train_Phi[i]) + lambd
            y_ij = y.get((i,j),0)
            obj += y_ij*cp.log(p) + (1-y_ij)*cp.log(1-p)
############################################################
#         P = np.matmul(train_Phi, w)
#         p = (1 - 2 * lambd) * P + lambd
#         obj = cp.sum(cp.multiply(y, cp.log(p)) + cp.multiply(1 - y, cp.log(1 - p)))
############################################################

        # Define the constraints
        constraints = [cp.sum(w) == 1, w >= 0]

        # Define the problem instance and solve
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver=cp.SCS, verbose=True, max_iters=300, eps=1e-6)
        
        # Retrieve argmax
        w_opt = w.value
        dict_w[j] = w_opt
#         print("node " + str(j) + " has w_opt " + str(w_opt))

#         f_jwlambda_S = lambd + (1-2*lambd)*np.dot(w_opt,X_Phi)
#         f.append(f_jwlambda_S)
    
    return dict_w

def influModelCompute(X,w_j,all_sinks,G,K,lambd):
    
    d = G.GetNodes()
    X_inc = incidence_conversion(X,d)
#     print(X_inc)
    f = []
    
    for j in range(d):
        X_Phi = Phi_vect(X_inc,all_sinks[j],K)
        f_jwlambda_S = lambd + (1-2*lambd)*np.dot(w_j[j],X_Phi)
        f.append(f_jwlambda_S)
        
    return sum(f)

def Du_sampleset_Worker(args):
    
    print("worker called....")
    mySeeds, G_nx, p, numCascadesPerSeed = args
    
    # convert G_nx back to snap graph
#     G_snap = snap.TNGraph.New()
#     for node in G_nx.nodes():
#         G_snap.AddNode(node)
#     for edge in G_nx.edges():
#         G_snap.AddEdge(edge[0], edge[1])
        
    mySamples = []
#     R_init = 0
    for seed in mySeeds:
        cascadeIdx = 0
        while cascadeIdx < numCascadesPerSeed: # Generate cascades for this seed
            activated = independent_cascade_deterministic(G_nx,p,seed)
            mySamples.append((seed,activated))
            cascadeIdx +=1
    print("worker done...")
    return mySamples

def independent_cascade_deterministic(G_nx, p, S):
    """
    Runs the independent cascade model on a directed graph G with edge probabilities p
    starting from a seed set S. Returns the set of activated nodes O.

    Args:
    - G: a directed networkx graph
    - p: a dictionary of edge weights in [0,1] (keys are (source, target) tuples, values are floats)
    - S: a set of node IDs representing the initial seed set
    - R: list of random numbers

    Returns:
    - O: a set of node IDs representing all the activated nodes throughout the algorithm
    """
    # Initialize the sets
    A = S.copy() # set of active nodes
    B = set()    # set of newly activated nodes
    O = S.copy()   # set of all activated nodes
    
    count=0
#     r = R_init
    while count < G_nx.number_of_nodes() and A:
        # Iterate over the active nodes
        for u in A:
            # Iterate over u's neighbors
            for v in G_nx.successors(u):
                # Check if v has already been activated
                if v in O:
                    continue
                # Check if the edge (u,v) is activated
                edge_prob = p.get((u, v), 0)
                #if R[r] < edge_prob:
                randy = random.random()
                # print(str(randy))
                if randy < edge_prob:
                    B.add(v)
                    O.add(v)
                #r+=1
        # Update the set of active nodes
        A = B.copy()
        B.clear()
        count+=1

    return O


# def independent_cascade_deterministic(G, p, S, R, R_init):
#     """
#     Runs the independent cascade model on a directed graph G with edge probabilities p
#     starting from a seed set S. Returns the set of activated nodes O.

#     Args:
#     - G: a directed graph (snap.TNGraph)
#     - p: a dictionary of edge weights in [0,1] (keys are (source, target) tuples, values are floats)
#     - S: a set of node IDs representing the initial seed set
#     - R: list of random numbers

#     Returns:
#     - O: a set of node IDs representing all the activated nodes throughout the algorithm
#     """
#     # Initialize the sets
#     A = S.copy() # set of active nodes
#     B = set()    # set of newly activated nodes
#     O = S.copy()   # set of all activated nodes
    
#     count=0
#     r = R_init
#     while count < G.GetNodes() and A:
#         # Iterate over the active nodes
#         for u in A:
#             # Iterate over u's neighbors
#             for v in G.GetNI(u).GetOutEdges():
#                 # Check if v has already been activated
#                 if v in O:
#                     continue
#                 # Check if the edge (u,v) is activated
#                 edge_prob = p.get((u, v), 0)
#                 if R[r] < edge_prob:
#                     B.add(v)
#                     O.add(v)
#                 r+=1
#         # Update the set of active nodes
#         A = B.copy()
#         B.clear()
#         count+=1

#     return (O,r)




############# IGNORE
# def kitchen_sink(j,q,K,d):
#     """
#     Generates a list of random binary feature vectors (See Sec 4. of Du)
    
#     Args:
#     -j: node of graph under consideration
#     -q: dictionary which each entry q_ij is the fraction of times that "node j in actived set" occurs over all samples where "node i was in seed"
#     -K: InfluLearner parameter
#     -d: dimension of graph, i.e. number of nodes in V
    
#     Returns:
#     - features: list of random vectors in {0,1}^d
    
#     """
#     features = [] # initialization
    
#     for i in range(K):
        
#         v = np.zeros(d)
#         for u in range(d):
            
#             # for each entry, with prob. q_uj make entry '1'
#             prob_u = q.get((u,j),0)
            
#             if random.random() < prob_u:
#                 v[u] = 1
                
#         # add vector to list, iterate
#         features.append(v)

#     return features
