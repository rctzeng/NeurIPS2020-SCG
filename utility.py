import optparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, norm
from scipy.stats import mode
from blist import sorteddict

DATASET_LIST = ['wow8', 'bitcoin', 'wikivot', 'referendum', 'slashdot', 'wikicon', 'epinions','wikipol','wikicon_aug','wikicon_noised_8']
ROUNDING_LIST = ['min_angle', 'randomized', 'max_obj', 'bansal']
FRAMEWORK_TYPE = ['recursive', 'flat'] # flat directly takes K-1 principal compoments
TARGET_MATRIX = ['adj', 'laplacian']
NORMALIZATION = ['none', 'sym', 'rw', 'sym_sep', 'rw_sep']
EPS=1E-10
INF=1E10

#################### Read Input/Arguments ####################
def parse_arg():
    """ utility function to parse argument from command line """
    parser = optparse.OptionParser()
    parser.add_option('-d', dest='dataset', default='all', help='specify the name of dataset in one of {}'.format(DATASET_LIST))
    parser.add_option('-k', dest='K', default=None, help='# of polarized communities')
    parser.add_option('-r', dest='rounding_strategy', default='min_angle', help='specify the rounding strategy in one of {}'.format(ROUNDING_LIST))
    parser.add_option('-n', dest='sbm_nv', default='1000', help='specify the graph size')
    parser.add_option('-c', dest='sbm_nc', default='50', help='specify the community size')
    (options, args) = parser.parse_args()
    return options

def read_graph(path):
    """ read the input graph and cast into scipy.sparse.csr_matrix format """
    with open(path, "r") as f:
        data = f.read()
    N = int(data.split('\n')[0].split(' ')[1])
    A = sp.lil_matrix((N,N), dtype='d')
    for line in data.split('\n')[1:]:
        es = line.split('\t')
        if len(es)!=3: continue
        A[int(es[0]),int(es[1])] = int(es[2])
        A[int(es[1]),int(es[0])] = int(es[2])
    return N, A.tocsr()

def gen_SBM(p, K, N, C_size):
    """ planted communities """
    prob = np.array([p, max(0,1-2*p), p])
    prob /= prob.sum()
    A = np.random.choice([1,0,-1], (N,N), p=prob) # noise
    A = np.sign((A + A.T)/2)
    for k in range(K):
        Z = []
        for i in range(k): Z += [np.random.choice([-1,0,1], (C_size,C_size), p=[1-p, p/2, p/2])]
        Z += [np.random.choice([1,0,-1], (C_size,C_size), p=[1-p, p/2, p/2])]
        for i in range(k+1,K): Z += [np.random.choice([-1,0,1], (C_size,C_size), p=[1-p, p/2, p/2])]
        Z = np.hstack(Z)
        A[k*C_size:(k+1)*C_size,:K*C_size] = Z.copy()
        A[:K*C_size,k*C_size:(k+1)*C_size] = Z.T.copy()
    return N, sp.csr_matrix(A.astype(float))

#################### Orthogonal Eigenvector of KI-1 ####################
def EigenDecompose_Core(K):
    """ return the spectrum of the (KI-1_{KxK}) matrix """
    U = [[np.sqrt(1.0/K) for i in range(K)]] # eigenvector
    D = [2-K] # eigenvalue
    for i in range(K-1):
        # eigenvector
        x = []
        for j in range(i): x += [0]
        x += [-(K-1-i)*np.sqrt(1.0/K)]
        for j in range(K-i-1): x += [np.sqrt(1.0/K)]
        # normalize
        s = np.sqrt(sum([j*j for j in x]))
        x = [j/s for j in x]
        U += [x]
        # eigenvalue
        D += [2]
    return np.array(D), np.array(U).T

############################## Compute Objectives ##############################
def compute_Obj(Y, A, K): # main objective
    """ sum_{j=2}^K of (Y_{:,j})^TA(Y_{:,j}) / ((Y_{:,j})^T(Y_{:,j})) """
    num, de = 0, 0
    for i in range(K-1):
        num += (Y[:,i+1].T).dot(A.dot(Y[:,i+1]))
        de += Y[:,i+1].T@Y[:,i+1]
    return (num / de)

def compute_RayleighsQuotient(Y, A):
    """ Rayleigh's quotient with the vector Y and input matrix A """
    return (Y.T)@(A.dot(Y)) / (Y.T@Y)

def compute_polarity(C, A, K, N):
    D,U = EigenDecompose_Core(K)
    U = U[:, D.argsort()]
    Y = np.zeros((N,K))
    for i,c in enumerate(C):
        if c>-1: Y[i,:] = U[c-1,:].copy()
    return Y, compute_Obj(Y, A, K)

def check_result_KCG(C, Y, A, N, K, run_time):
    """ inspect the found K polarized communities """
    lD, _ = eigsh(A, k=1, which='LA')
    sD, _ = eigsh(A, k=1, which='SA')
    X = sp.lil_matrix((N,K), dtype='d')
    for i,c in enumerate(C):
        if c>-1: X[i,c-1] = 1
    X = X.tocsr()
    print('Obj = {:.1f} in ({:.1f}, {:.1f}), Execution Time={:.1f}'.format(compute_Obj(Y, A, K), sD[0], lD[0], run_time))
    n_in, n_out = 0, 0
    for k in range(K):
        nk = X[:,k].sum()
        if nk==0: continue
        In_n = ((np.abs(A)-A)/2.0).multiply(X[:,k]*(X[:,k].T)).sum()
        In_p = ((np.abs(A)+A)/2.0).multiply(X[:,k]*(X[:,k].T)).sum()
        Out_n = ((np.abs(A)-A)/2.0).multiply((X[:,:k-1].sum(axis=1)+X[:,k+1:].sum(axis=1))*(X[:,k].T)).sum()
        Out_p = ((np.abs(A)+A)/2.0).multiply((X[:,:k-1].sum(axis=1)+X[:,k+1:].sum(axis=1))*(X[:,k].T)).sum()
        print('|S_{}|={:.0f}, |In_+|-|In_-|={:.0f}-{:.0f}, |Out_-|-|Out_+|={:.0f}-{:.0f}'.format(
            k+1, nk, In_p, In_n, Out_n, Out_p
        ))
        n_in, n_out = n_in+In_p-In_n, n_out+Out_p-Out_n
        for j in range(K):
            if j==k: continue
            nj = X[:,j].sum()
            if nj==0: continue
    print('|S_0|={:.0f} // neutral'.format(N-X.sum()))
    print('Total: |S_1|+...+|S_K|={:.0f}, |In_+|-|In_-|={:.0f}, |Out_+|-|Out_-|={:.0f}'.format(X.sum(), n_in, n_out))
    print('---------------------------')

def compute_accuracy(C, nC, K, EPS=1E-10):
    """ compute accuracy of the SBM model """
    modes = []
    for i in range(K):
        values = [x for x in C[i*nC:(i+1)*nC] if x != -1]
        if values == []: modes += [-1]
        else: modes += [mode(values)[0][0]]
    pred_total, pred_correct, rec_num = [0 for i in range(K)], [0 for i in range(K)], [0 for i in range(K)]
    for i in range(len(C)):
        if C[i]>0: pred_total[C[i]-1] += 1
        k = int(i/nC)
        if k<K and i<nC*K and C[i]==modes[k] and C[i]>0:
            pred_correct[C[i]-1] += 1
            rec_num[C[i]-1] += 1
    precs = [pred_correct[i]/(pred_total[i]+EPS) for i in range(K) if pred_total[i]>0]
    recs = [min(nC,rec_num[i])/nC for i in range(K)]
    prec = np.mean(precs)
    rec = np.mean(recs)
    f1_score = 2*prec*rec/(prec+rec+EPS)
    return prec, rec, precs, recs, f1_score

def compute_accuracy_Kplus1(C, nC, K, EPS=1E-10):
    """ compute accuracy of the SBM model """
    modes = []
    for i in range(K):
        values = [x for x in C[i*nC:(i+1)*nC] if x != -1]
        if values == []: modes += [-1]
        else: modes += [mode(values)[0][0]]
    pred_total, pred_correct, rec_num = [0 for i in range(K+1)], [0 for i in range(K+1)], [0 for i in range(K+1)]
    for i in range(len(C)):
        if C[i]>0: pred_total[C[i]-1] += 1
        k = int(i/nC)
        if k<K and i<nC*K and C[i]==modes[k] and C[i]>0:
            pred_correct[C[i]-1] += 1
            rec_num[C[i]-1] += 1
    idx = np.argsort(pred_correct)[::-1]
    precs = [pred_correct[idx[i]]/(pred_total[idx[i]]+EPS) for i in range(K) if pred_total[idx[i]]>0]
    recs = [min(nC,rec_num[idx[i]])/nC for i in range(K)]
    prec = np.mean(precs)
    rec = np.mean(recs)
    f1_score = 2*prec*rec/(prec+rec+EPS)
    return prec, rec, precs, recs, f1_score, idx[:K]

############################## Rounding Algorithms ##############################
#### [Deterministic] Minimum Angle ####
def min_angle_find_k1_k2(v, idx, pos, neg, N):
    """ find the two threshold to round the given vector v """
    def next_move(v, T, i, j):
        distances = []
        for ci,cj in [(1,0),(0,-1)]: # no go back
            cT = T.copy()
            ti, tj = i+ci, j+cj
            i_invalid, j_invalid = (ti<0 or ti>=N), (tj<0 or tj>=N)
            if i_invalid and j_invalid: continue
            if i_invalid and cj==0: continue
            if j_invalid and ci==0: continue
            if not i_invalid and ci==1: cT[idx[ti]] = pos
            elif not j_invalid and  cj==-1: cT[idx[tj]] = neg
            e = cT/np.linalg.norm(cT)
            diff = np.linalg.norm(v - v.dot(e)*e)
            distances += [(ti,tj,diff,cT.copy())]
        # pick the best for the next move
        min_idx = np.argmin([x[2] for x in distances])
        return distances[min_idx]
    k1, k2, dist_opt, T_opt = -1, N, INF, np.zeros((N))
    while k1<k2:
        k1_next, k2_next, dist_next, T_next = next_move(v, T_opt, k1, k2)
        if dist_next >= dist_opt: break
        k1, k2, dist_opt, T_opt = k1_next, k2_next, dist_next, T_next.copy()
    return T_opt, dist_opt, k1, k2

def round_by_min_angle(v, pos, neg, mask, N):
    """ find the vector T in {0,-1,z}^N with minimum angle to the vector v or -v """
    # (1) sort v by its magnitude in the non-increasing order
    v = v*mask # skip assigned nodes
    idx_x = np.argsort(v, axis=0)[::-1]
    idx_y = idx_x[::-1]
    # (2) find the v's closest projection in {0,-1,z}^N
    x, x_diff, x_k1, x_k2 = min_angle_find_k1_k2(v, idx_x, pos, neg, N)
    y, y_diff, y_k1, y_k2 = min_angle_find_k1_k2(-v, idx_y, pos, neg, N)
    v_round = x if x_diff < y_diff else y
    return v_round

#### [Deterministic] Maximum Objective ####
def round_by_max_obj_one_threshold(v_in, pos, neg, mask, A, N):
    """ find the threshold to round the vector v or -v """
    def max_obj_find_th(v):
        """ find the two threshold to round the given vector v """
        T_opt, obj_opt, th_opt = np.zeros(N), 0, None
        for th in set([int(abs(e) * 1000) / 1000.0 for e in v]):
            T = pos*(v>0)*(np.abs(v)>=th)+neg*(v<0)*(np.abs(v)>=th)
            if np.sum(T)==0: continue
            obj = compute_RayleighsQuotient(T, A)
            if obj>obj_opt: T_opt, obj_opt, th_opt = T.copy(), obj, th
        return T_opt, obj_opt, th_opt
    v_in = v_in*mask # skip assigned nodes
    x, x_obj, x_th = max_obj_find_th(v_in)
    y, y_obj, y_th = max_obj_find_th(-v_in)
    v_round = x if x_obj > y_obj else y
    return v_round.copy()

#### [Randomized] ####
def round_by_randomized_vector(v_in, pos, neg, mask, A, N):
    """ sample a randomized vector T in {0,-1,z}^N given the specific vector v or -v """
    def randomized_vector(v):
        def bernoulli_sample(x):
            if x>0: return pos*np.random.choice([0,1], 1, p=[max(1-x/pos,0),min(x/pos,1)])[0]
            elif x<0: return neg*np.random.choice([0,1], 1, p=[max(1-x/neg,0),min(x/neg,1)])[0]
            else: return 0
        v *= np.abs(v).sum()
        T = np.array([bernoulli_sample(v[i]) for i in range(N)])
        return T
    v_in = v_in*mask # skip assigned nodes
    x = randomized_vector(v_in)
    y = randomized_vector(-v_in)
    v_round = x.copy() if compute_RayleighsQuotient(x, A)>compute_RayleighsQuotient(y, A) else y.copy()
    return v_round

#### [Correlation Clustering: Bansal 3-approximation] ####
def round_by_cc_bansal(pos, neg, mask, A, N):
    """ identify 2 communities by splitting the neighborhood of 1 node """
    def find_one_neighborhood_split():
        T, obj_opt = None, None
        for i in range(N):
            _, nbrs = A[i,:].nonzero()
            S1, S2 = [], []
            for  j in nbrs:
                if A[i,j]>0: S1 += [j]
                elif A[i,j]<0: S2 += [j]
            # decide how (S1, S2) should be associate with -1 and q
            T1, T2 = np.zeros(N), np.zeros(N)
            for j in S1: T1[j], T2[j] = pos, neg
            for j in S2: T1[j], T2[j] = neg, pos
            obj1, obj2 = compute_RayleighsQuotient(T1, A), compute_RayleighsQuotient(T2, A)
            T_i, obj_i = (T1, obj1) if obj1>obj2 else (T2, obj2)
            if obj_opt==None or obj_i>obj_opt:
                T, obj_opt = T_i.copy(), obj_i
        return T
    v_round = find_one_neighborhood_split()
    return v_round
