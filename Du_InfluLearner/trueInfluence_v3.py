def compute_true_infl(args):
    sub_task, S, C, T = args
    true_infl = 0
    idx = 0
    
    for j in range(sub_task):
        F = set()
        for s in S:
            saved_rand_int = T[s][j]
            F |= C[s][saved_rand_int]
            
        true_infl += len(F)
        
    return true_infl/sub_task