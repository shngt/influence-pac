import snap
import random
import numpy as np
import torch
import itertools
import sys
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample_power_law(U):
    # Sample the size of the set according to a power law distribution
    size = int(pow(random.random(), -1/2.5))
    # Sample elements from U without replacement
    return set(random.sample(U, size))

def independent_cascade(G, p, S):
    """
    Runs the independent cascade model on a directed graph G with edge probabilities p
    starting from a seed set S. Returns a list of the form [((V, v), y)_t] which .

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
    F = S.copy() # set of all activated nodes
    O = []       # Output list 
    
    count = 0
    while len(A) > 0:
        # Iterate over the active nodes
        # print("iteration : " + str(count))
        # print("active nodes: " + str(A))
        O_t = []
        for u in A:
            # print("consider active node " + str(u))
            # Iterate over u's neighbors
            for v in G.GetNI(u).GetOutEdges():
                # print(str(u) + " has out-neighbor " + str(v))
                # Check if v has already been activated
                if v in F:
                    # print(str(v) + " has already been activated")
                    O_t.append((A, v, 0))
                    continue
                # Check if the edge (u,v) is activated
                edge_prob = p.get((u, v), 0)
                # print("p(u,v) is " + str(p.get((u, v), 0)))
                if random.random() < edge_prob:
                    # print(str(v) +" was succesfully infected")
                    B.add(v)
                    F.add(v)
                    O_t.append((A, v, 1))
                else:
                    O_t.append((A, v, 0))
        # Update the set of active nodes
        A = B.copy()
        if len(O_t) > 0:
            O.append(O_t)
        B.clear()
        count += 1

    return O

def independent_cascade_km(G, p, S):
    """
    Runs the independent cascade model on a directed graph G with edge probabilities p
    starting from a seed set S. Returns a list of the form [((V, v), y)_t] which .

    Args:
    - G: a directed graph (snap.TNGraph)
    - p: a dictionary of edge probabilities (keys are (source, target) tuples, values are floats)
    - S: a set of node IDs representing the initial seed set

    Returns:
    - O: a set of node IDs representing all the activated nodes throughout the algorithm
    """
    # Initialize the sets
    A = S.copy() # set of active nodes
    O = []       # Output list 
    
    # Iterate over the active nodes
    # print("iteration : " + str(count))
    # print("active nodes: " + str(A))
    for u in A:
        # print("consider active node " + str(u))
        # Iterate over u's neighbors
        for v in G.GetNI(u).GetOutEdges():
            # print(str(u) + " has out-neighbor " + str(v))
            # Check if v has already been activated
            # Check if the edge (u,v) is activated
            edge_prob = p.get((u, v), 0)
            # print("p(u,v) is " + str(p.get((u, v), 0)))
            if random.random() < edge_prob:
                # print(str(v) +" was succesfully infected")
                O.append((A, v, 1))
            else:
                O.append((A, v, 0))

    return O

def gen_IC_cascades(G, probabilities, num_samples):
    training_examples =[]
    num_samples_so_far = 0
    while num_samples_so_far < num_samples:
        seed_set = sample_power_law(set(range(G.GetNodes()))) # sample_bernoulli(range(G.GetNodes())) 
        example = independent_cascade_km(G, probabilities, seed_set)
        if len(example) > 0:
            num_remaining = min(num_samples - num_samples_so_far, len(example))
            training_examples.append(example[:num_remaining])
            # print(example)
        num_samples_so_far += len(example[:num_remaining])
        # print(num_samples_so_far)
    return training_examples

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def genRand_edge_probabilities(G, x_V, theta, D_i):
    probabilities = {}
    for edge in G.Edges():
        u = edge.GetSrcNId()
        v = edge.GetDstNId()
        x_u, x_v = x_V[u], x_V[v]
        x_uv = np.concatenate([x_u[:D_i], x_v[:D_i]])
        p_uv = sigmoid(np.dot(x_uv, theta))
        probabilities[(u, v)] = p_uv
    return probabilities

def learn_edge_probs(G, x_V, training_samples, D_g):
    x_V = torch.as_tensor(x_V, device=device, dtype=torch.float)
    theta = torch.zeros(D_g, requires_grad=True, device=device, dtype=torch.float)
    max_epochs, lr, threshold = 10, (1 / len(training_samples)) ** 0.5, 1e-4
    complete = False
    while not complete and max_epochs > 0:
        random.shuffle(training_samples)
        for sample in tqdm(training_samples):
            X, v, y = sample
            # print(sample)
            X_uv = torch.zeros((0, D_g), device=device, dtype=torch.float)
            for x in X:
                if x in G.GetNI(v).GetInEdges():
                    # print(x, 'is a valid in-neighbour')
                    x_uv = torch.unsqueeze(torch.cat((x_V[x], x_V[v])), 0)
                    X_uv = torch.cat((X_uv, x_uv))
            assert X_uv.shape[0] > 0
            # if X_uv.shape[0] > 1:
            #     import pdb; pdb.set_trace()
            # print(X_uv.shape)
            p = torch.matmul(X_uv, theta)
            # print('p', p.shape, p)
            q = torch.sigmoid(p)
            # print('q', q.shape, q)
            r = torch.prod(1 - q) + 1e-6
            f = 1 - r
            # print('f', f)
            nll = -1 * (y * torch.log(f) + (1 - y) * torch.log(1 - f))
            # print('nll', nll)
            nll.backward()

            if torch.linalg.vector_norm(lr * theta.grad) < threshold:
                complete = True
                break

            with torch.no_grad():
                theta -= lr * theta.grad 
            theta.grad.zero_()
        
        max_epochs -= 1
            # print('theta', theta)
    return theta

def evaluator(G, true_weights, pred_weights, num_samples):
    ae = 0
    testing_samples = gen_IC_cascades(G, true_weights, num_samples)
    for sample in testing_samples:
        num_act = sum(s[2] for s in sample)
        gen_sample = independent_cascade_km(G, pred_weights, sample[0][0])
        num_act_gen = sum(s[2] for s in gen_sample)
        ae += abs(num_act - num_act_gen)
    mae = ae / len(testing_samples)
    return mae

def main(V, E, M):
    # V, E = 200, 400
    D_a, D_i, D_g = 10, 10, 20
    rnd = snap.TRnd(42)
    G = snap.GenRndGnm(snap.TNGraph, V, E, True, rnd)
    x_V = np.array([np.random.rand(D_a) for v in range(V)])
    theta = np.random.rand(D_g) * 2 - 1
    true_weights = genRand_edge_probabilities(G, x_V, theta, D_i)
    # print(true_weights)
    training_samples = list(itertools.chain.from_iterable(gen_IC_cascades(G, true_weights, M)))
    # print(training_samples)
    # testing_samples = gen_Uniform_IC_Cascades(G, true_weights, 16, 1000)
    # print(testing_samples)
    print('Training samples generated')
    pred_theta = learn_edge_probs(G, x_V, training_samples, D_g)
    print('true theta: ', theta)
    print('pred theta: ', pred_theta)
    pred_weights = genRand_edge_probabilities(G, x_V, theta, D_i)
    mae = evaluator(G, true_weights, pred_weights, 100000)
    print('mae', mae)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('4 arguments needed - V, E, number of samples M and number of trials')
        exit(0)
    V, E, M, trials = list(map(int, sys.argv[1:]))
    for seed in range(trials):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        main(V, E, M)