"""
Related work: "Finding Gangs in War from Signed Networks", KDD'16
This is a Python wrapper to run the Matlab code provided by the authors (repo link: https://github.com/lingyangchu/KOCG.SIGKDD2016)

Requirement:
 * Install python package `matlab_engine` following the link: https://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
"""
import os
import glob
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import matlab.engine
import sys
import re
sys.path.append('../')
from utility import *

def parse_arg():
    """ utility function to parse argument from command line """
    parser = optparse.OptionParser()
    parser.add_option('-d', dest='dataset', default='all', help='specify the name of dataset in one of {}'.format(DATASET_LIST))
    parser.add_option('-k', dest='K', default=None, help='# of polarized communities')
    parser.add_option('-n', dest='sbm_nv', default='1000', help='specify the graph size')
    parser.add_option('-c', dest='sbm_nc', default='50', help='specify the community size')
    parser.add_option('-t', dest='topr', default='Topr', help='specify to choose Top1 or Topr ranking')
    parser.add_option('-f', dest='mode', default='Eval', help='specify to Run or Eval')
    (options, args) = parser.parse_args()
    return options

def init_graph(name):
    """ process graphs to be matlab to compare with baselines """
    if not os.path.exists('datasets/{}.mat'.format(name)):
        # read graph in scipy.sparse format
        with open("../datasets/{}.txt".format(name), "r") as f:
            data = f.read()
        N = int(data.split('\n')[0].split(' ')[1])
        A = sp.lil_matrix((N,N), dtype='d')
        for line in data.split('\n')[1:]:
            es = line.split('\t')
            if len(es)!=3: continue
            A[int(es[0]),int(es[1])] = int(es[2])
            A[int(es[1]),int(es[0])] = int(es[2])
        # convert to .mat
        sio.savemat('../datasets/{}.mat'.format(name), {'A':A})

def init_graph_sbm(t, p, K, N, C_size):
    """ process graphs to be matlab to compare with baselines """
    if not os.path.exists('../datasets/sbm_t{}_p{}.mat'.format(t, int(p*10))):
        N, A = gen_SBM(p, K, N, C_size)
        A = A.tolil().astype(dtype='d')
        sio.savemat('../datasets/sbm_t{}_p{}.mat'.format(t, int(p*10)), {'A':A})
        with open("../datasets/sbm_t{}_p{}.txt".format(t, int(p*10)), "w") as f:
            f.write('# {}\n'.format(N))
            A = A.tocoo()
            for i,j,v in zip(A.row, A.col, A.data):
                f.write('{}\t{}\t{}\n'.format(i,j,int(v)))

def check_result_fout(C, S, A, N, K, f):
    X = sp.lil_matrix((N,K), dtype='d')
    for i,c in enumerate(C):
        if c>-1: X[i,c-1] = 1
    X = X.tocsr()
    _, obj = compute_polarity(C, A, K, N)
    f.write('Obj = {:.1f}\n'.format(obj))
    for k in range(K):
        nk = X[:,k].sum()
        if nk==0: continue
        In_n = ((np.abs(A)-A)/2.0).multiply(X[:,k]*(X[:,k].T)).sum()
        In_p = ((np.abs(A)+A)/2.0).multiply(X[:,k]*(X[:,k].T)).sum()
        Out_n = ((np.abs(A)-A)/2.0).multiply((X[:,:k-1].sum(axis=1)+X[:,k+1:].sum(axis=1))*(X[:,k].T)).sum()
        Out_p = ((np.abs(A)+A)/2.0).multiply((X[:,:k-1].sum(axis=1)+X[:,k+1:].sum(axis=1))*(X[:,k].T)).sum()
        f.write('|S_{}|={:.0f}, |In_+|-|In_-|={:.0f}-{:.0f}, |Out_-|-|Out_+|={:.0f}-{:.0f}\n'.format(
            k+1, X[:,k].sum(), In_p, In_n, Out_n, Out_p
        ))
        for j in range(K):
            if j==k: continue
            nj = X[:,j].sum()
            if nj==0: continue
    f.write('|S_0|={:.0f} // neutral\n'.format(N-X.sum()))
    f.write('---------------------------\n')

DATASET_LIST = ['wow8', 'bitcoin', 'wikivot', 'referendum', 'slashdot', 'wikicon', 'epinions','wikipol']
opt = parse_arg()
TYPE = opt.topr # ['Top1', 'Topr']
MODE = opt.mode # ['Run', 'Eval']
if 'Eval' in MODE:
    if opt.K == None: raise Exception('Error: please specify K')
    try: K = int(opt.K)
    except ValueError: raise Exception('Error: please specify K in integer')

if opt.dataset != 'sbm':
    # prepare input in .mat for .m script
    for name in DATASET_LIST: init_graph(name)
    if 'Run' in MODE:
        for k in [2,6]:
            if not os.path.exists('K{}'.format(k)): os.makedirs('K{}'.format(k))
        # run .m script
        eng = matlab.engine.start_matlab()
        eng.runKOCG_datasets(nargout=0)
    if 'Eval' in MODE:
        # read results: Top1(in KOCG's ranking) and Topr(select size equals to ours)
        with open('../result/results_K{}_SCG-MA.txt'.format(K)) as f:
            data = f.read()
        my_covered = {d:int(x) for d,x in zip(re.findall('------ Running (.*).txt ------\n', data),
            re.findall('\|S\_0\|=(.*) \/\/ neutral\n', data))}
        # read-out the resulted clusters
        f = open('../result/results_K{}_KOCG_{}.txt'.format(K, TYPE), 'w')
        for name in DATASET_LIST:
            f.write('------ Running {}.txt ------\n'.format(name))
            print(name)
            N, A = read_graph("../datasets/{}.txt".format(name))
            C = [-1 for i in range(N)]
            # pick p to cover similar ratio of nodes
            S, cnt = [], 0
            fs = glob.glob('K{}/result_{}_p*.mat'.format(K, name))
            for i in range(len(fs)):
                cX = sio.loadmat('K{}/result_{}_p{}.mat'.format(K, name, i+1))
                X = cX['X'].tocoo()
                cur = []
                for r,c,v in zip(X.row,X.col,X.data):
                    if TYPE == 'Topr' and cnt>=(N-my_covered[name]): break
                    if C[r]==-1:
                        C[r] = c+1
                        cnt += 1
                    cur += [r]
                S += [cur]
                if TYPE == 'Top1': break
            check_result_fout(C, S, A, N, K, f)
        f.close()
else:
    try: N = int(opt.sbm_nv)
    except ValueError: raise Exception('Error: please specify the graph size in integer')
    try: nC = int(opt.sbm_nc)
    except ValueError: raise Exception('Error: please specify the community size in integer')
    if 'Run' in MODE:
        # generate sbm graphs
        for t in range(20):
            for p in [0.1*(i) for i in range(7)]:
                init_graph_sbm(t, p, 6, N, nC)
        if not os.path.exists('K6'): os.makedirs('K6')
        # run .m code
        eng = matlab.engine.start_matlab()
        eng.runKOCG_sbm(nargout=0)
    if 'Eval' in MODE:
        # read results: Top1(in KOCG's ranking) and Topr(select size equals to ours)
        f = open('../result/sbm_K{}_KOCG_{}.txt'.format(K, TYPE), 'w')
        for t in range(20):
            f.write('------------ [Round #{}] ------------\n'.format(t))
            for p in [0.1*(i) for i in range(7)]:
                f.write('------ Running SBM [p={:.1f}] ------\n'.format(p))
                print(p)
                N, A = read_graph("../datasets/sbm_t{}_p{}.txt".format(t, int(p*10)))
                C = [-1 for i in range(N)]
                # pick p to cover similar ratio of nodes
                S, cnt = [], 0
                fs = glob.glob('K{}/sbm_t{}_p{}_p*.mat'.format(K, t, int(p*10)))
                for i in range(len(fs)):
                    cX = sio.loadmat('K{}/sbm_t{}_p{}_p{}.mat'.format(K, t, int(p*10), i+1))
                    X = cX['X'].tocoo()
                    cur = []
                    for r,c,v in zip(X.row,X.col,X.data):
                        if TYPE == 'Topr' and cnt>=(K*nC): break
                        if C[r]==-1:
                            C[r] = c+1
                            cnt += 1
                        cur += [r]
                    S += [cur]
                    if TYPE == 'Top1': break
                check_result_fout(C, S, A, N, K, f)
                precision, recall, precs, recs, f1_score = compute_accuracy(C, nC, K)
                f.write('Accuracy: precision={:.2f}, recall={:.2f}, f1-score={:.2f}\n'.format(precision, recall, f1_score))
                f.write('{}\n'.format(precs))
                f.write('{}\n'.format(recs))
        f.close()
