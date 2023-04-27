import snap
import random
from math import pow
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import multiprocessing as mp
import sys

def sample_power_law(U):
    # Sample the size of the set according to a power law distribution
    size = int(pow(random.random(), -1/0.6))
    print(size)
    # Sample elements from U without replacement
    return set(random.sample(U, size))

def sample_bernoulli(U):
    seed_set = set()
    for u in U:
        if random.random() < 0.02:
            seed_set.add(u)
    return seed_set

def independent_cascade(G, p, S):
    """
    Runs the independent cascade model on a directed graph G with edge probabilities p
    starting from a seed set S. Returns the set of activated nodes O.

    Args:
    - G: a directed graph (snap.TNGraph)
    - p: a dictionary of edge probabilities (keys are (source, target) tuples, values are floats)
    - S: a set of node IDs representing the initial seed set

    Returns:
    - O: a set of node IDs representing all the activated nodes throughout the algorithm
    """
    # Initialize the sets
    A = S.copy() # set of active nodes
    B = set()    # set of newly activated nodes
    O = S.copy()   # set of all activated nodes
    T = {s: 0 for s in A} # Time of infection
    
    count=0
    while count < G.GetNodes() and len(A) > 0:
        # Iterate over the active nodes
        # print("iteration : " + str(count))
        # print("active nodes: " + str(A))
        for u in A:
            # print("consider active node " + str(u))
            # Iterate over u's neighbors
            for v in G.GetNI(u).GetOutEdges():
                # print(str(u) + " has out-neighbor " + str(v))
                # Check if v has already been activated
                if v in O:
                    # print(str(v) + " has already been activated")
                    continue
                # Check if the edge (u,v) is activated
                edge_prob = p.get((u, v), 0)
                # print("p(u,v) is " + str(p.get((u, v), 0)))
                if random.random() < edge_prob:
                    # print(str(v) +" was succesfully infected")
                    B.add(v)
                    O.add(v)
                    T[v] = count + 1
        # Update the set of active nodes
        A = B.copy()
        B.clear()
        count+=1

    return (O, T)

def gen_Uniform_IC_Cascades(G, probabilities, numSamples, samples_per_seed):
    training_examples = []
    count = 0
    while count < numSamples:
        seed_set = sample_bernoulli(range(G.GetNodes())) # sample_power_law(set(range(G.GetNodes()))) # 
        print(seed_set)
        for _ in range(samples_per_seed):
            (activated, times) = independent_cascade(G, probabilities, seed_set)
            example = (seed_set, activated, times)
            training_examples.append(example)
        count+=1
    return training_examples

def genRand_edge_probabilities(G):
    probabilities = {}
    for edge in G.Edges():
        u = edge.GetSrcNId()
        v = edge.GetDstNId()
        weight = random.random()
        probabilities[(u, v)] = weight
    return probabilities

def learn_edge_probabilities_cp(v, training_samples, V):
    def loss(w, T_na, T_ac):
        L = 0.0
        w_na = cp.multiply(w, T_na) # Mask out weights
        w_ac = cp.multiply(w, T_ac)
        L -= cp.sum(-1 * w_na + cp.log(1 - cp.exp(-1 * w_ac - 1e-4)))
        return L

    # Solve optimization problem for each node i separately
    # bnds = [(0, None) for _ in range(V)]
    T_na = np.zeros(V)
    T_ac = np.zeros(V)

    for s_i, sample in enumerate(training_samples):
        S, O, T = sample
        t_v = T.get(v, -1)
        for u, t_u in T.items():
            if t_u <= t_v - 2:
                T_na[u] += 1
            elif t_u == t_v - 1:
                T_ac[u] += 1

    w = cp.Variable(V)
    obj = cp.Minimize(loss(w, T_na, T_ac))
    constraints = [0 <= w]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, max_iters=500)
    except:
        print('Failed')
        print(w, T_na, T_ac)
        import pdb; pdb.set_trace()
    thetas = w.value
    probs = 1 - np.exp(-1 * thetas)
    print(v, 'Node done')
    return probs

def learn_edge_probabilities(v, training_samples, V):
    # print(v, 'Node started')
    def loss(w, T_na, T_ac):
        L = 0.0
        w_na = np.multiply(w, T_na) # Mask out weights
        w_ac = np.multiply(w, T_ac)
        L -= np.sum(-1 * np.sum(w_na, axis=1) + np.log(1 - np.exp(-1 * np.sum(w_ac, axis=1) - 1e-4)))
        return L
    # Solve optimization problem for each node i seperately
    bnds = [(0, None) for _ in range(V)]
    T_na = np.zeros((len(training_samples), V))
    T_ac = np.zeros((len(training_samples), V))
    # print(training_samples)
    for s_i, sample in enumerate(training_samples):
        S, O, T = sample
        t_v = T.get(v, -1)
        for u, t_u in T.items():
            if t_u <= t_v - 2:
                T_na[s_i][u] = 1
            elif t_u == t_v - 1:
                T_ac[s_i][u] = 1
    w0 = np.zeros(V) 
    res = minimize(loss, w0, args=(T_na, T_ac), bounds=bnds)
    thetas = res.x
    probs = 1 - np.exp(-1 * thetas)
    # print(v, 'Node done')
    return probs

def learn_edge_probabilities_v2(v, training_samples, V):
    def loss(w, T_na, T_ac):
        L = 0.0
        w_na = np.multiply(w, T_na) # Mask out weights
        print(w_na, T_na)
        w_ac = np.multiply(w, T_ac)
        L -= np.sum(-1 * w_na + np.log(1 - np.exp(-1 * w_ac - 1e-4)))
        return L
    # Solve optimization problem for each node i seperately
    bnds = [(0, None) for _ in range(V)]
    T_na = np.zeros(V)
    T_ac = np.zeros(V)
    # print(training_samples)
    for s_i, sample in enumerate(training_samples):
        S, O, T = sample
        t_v = T.get(v, -1)
        for u, t_u in T.items():
            if t_u <= t_v - 2:
                T_na[u] += 1
            elif t_u == t_v - 1:
                T_ac[u] += 1
    w0 = np.zeros(V) 
    res = minimize(loss, w0, args=(T_na, T_ac), bounds=bnds)
    thetas = res.x
    probs = 1 - np.exp(-1 * thetas)
    # print(v, 'Node done')
    return probs

def evaluator(G, pred_weights, testing_samples, num_seeds, sample_per_seed):
    avg_pred_infls = []
    avg_true_infls = []
    for seed_idx in range(num_seeds):
        pred_infl = 0.0
        true_infl = 0.0
        for sample_idx in range(sample_per_seed):
            idx = seed_idx * sample_per_seed + sample_idx
            # print('test seed', testing_samples[idx][0])
            # print('test_sample', testing_samples[idx])
            # print('influenced_nodes', independent_cascade(G, pred_weights, testing_samples[idx][0]))
            pred_infl += len(independent_cascade(G, pred_weights, testing_samples[idx][0])[0])
            true_infl += len(testing_samples[idx][1])
        pred_infl /= sample_per_seed
        true_infl /= sample_per_seed
        avg_pred_infls.append(pred_infl)
        avg_true_infls.append(true_infl)
    # print(avg_pred_infls, avg_true_infls)
    # from scipy.stats.mstats import winsorize
    maes = []
    print(avg_pred_infls, avg_true_infls)
    for p_i, t_i in zip(avg_pred_infls, avg_true_infls):
        maes.append(abs(p_i - t_i))
    # print(maes)
    # print('Before Winsorizing: ', len(maes))
    #maes = winsorize(maes, limits=[0.1, 0.1])
    # print('After Winsorizing: ', len(maes))
    mae = sum(maes) / len(maes)
    return mae        

def adj_matrix_to_dict(pred_weights):
    sparse_pred_weights = sparse.csr_matrix(pred_weights)
    return sparse_pred_weights.todok()

def main(V, E, M):
    # V, E = 1000, 20000
    rnd = snap.TRnd(42)
    G = snap.GenRndGnm(snap.TNGraph, V, E, True, rnd)
    true_weights = genRand_edge_probabilities(G)
    training_samples = gen_Uniform_IC_Cascades(G, true_weights, M, 100)
    testing_samples = gen_Uniform_IC_Cascades(G, true_weights, M, 100)
    #print(testing_samples)
    print('Training samples generated')
    pred_weights = []
    # for v in range(V):
    #   pred_weights.append(learn_edge_probabilities(v, training_samples, V))
    with mp.Pool(processes=32) as pool:
        pred_weights = pool.starmap(learn_edge_probabilities, [(v, training_samples, V) for v in range(V)])
    pred_weights = np.array(pred_weights).T
    pred_weights_dict = adj_matrix_to_dict(pred_weights)
    print(true_weights)
    print(pred_weights_dict)
    mae = evaluator(G, pred_weights_dict, testing_samples, M, 100)
    print('mae', mae)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('4 arguments needed - V, E, number of samples M and number of trials')
        exit(0)
    V, E, M, trials = list(map(int, sys.argv[1:]))
    for seed in range(trials):
        random.seed(seed)
        np.random.seed(seed)
        main(V, E, M)