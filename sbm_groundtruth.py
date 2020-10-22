import os
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import sys
import re
from utility import *

def parse_arg():
    """ utility function to parse argument from command line """
    parser = optparse.OptionParser()
    parser.add_option('-k', dest='K', default=None, help='# of polarized communities')
    parser.add_option('-n', dest='sbm_nv', default='2000', help='specify the graph size')
    parser.add_option('-c', dest='sbm_nc', default='100', help='specify the community size')
    (options, args) = parser.parse_args()
    return options

def check_result_fout(C, A, N, K, f):
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

opt = parse_arg()

try: K = int(opt.K)
except ValueError: raise Exception('Error: please specify the number of group in integer')
try: N = int(opt.sbm_nv)
except ValueError: raise Exception('Error: please specify the graph size in integer')
try: nC = int(opt.sbm_nc)
except ValueError: raise Exception('Error: please specify the community size in integer')

f = open('result/sbm_K{}_GroundTruth.txt'.format(K), 'w')
for t in range(20):
    f.write('------------ [Round #{}] ------------\n'.format(t))
    for p in [0.1*(i) for i in range(7)]:
        f.write('------ Running SBM [p={:.1f}] ------\n'.format(p))
        print(p)
        N, A = read_graph("datasets/sbm_t{}_p{}.txt".format(t, int(p*10)))
        C = [int(i/nC)+1 for i in range(nC*K)]+[-1 for i in range(N-nC*K)]
        check_result_fout(C, A, N, K, f)
        precision, recall, precs, recs, f1_score = compute_accuracy(C, nC, K)
        f.write('Accuracy: precision={:.2f}, recall={:.2f}, f1-score={:.2f}\n'.format(precision, recall, f1_score))
        f.write('{}\n'.format(precs))
        f.write('{}\n'.format(recs))
f.close()
