from signet.cluster import Cluster
from scipy.sparse import eye
from utility import *
import time

"""
Requirement
 * `pip install git+https://github.com/alan-turing-institute/SigNet.git`
"""

DATASET_LIST = ['wow8', 'bitcoin', 'wikivot', 'referendum'] # ['slashdot', 'wikicon', 'epinions','wikipol']
METHOD = ['bnc-sym', 'sponge', 'sponge-sym']

def parse_arg():
    """ utility function to parse argument from command line """
    parser = optparse.OptionParser()
    parser.add_option('-d', dest='dataset', default='all', help='specify the name of dataset in one of {}'.format(DATASET_LIST))
    parser.add_option('-k', dest='K', default=None, help='# of polarized communities')
    parser.add_option('-m', dest='method', default='bnc-sym', help='specify the baseline method in one of {}'.format(METHOD))
    parser.add_option('-n', dest='sbm_nv', default='1000', help='specify the graph size')
    parser.add_option('-c', dest='sbm_nc', default='50', help='specify the community size')
    (options, args) = parser.parse_args()
    return options

def run(dataset, K, method, N=None, A=None):
    """ find K polarized communities """
    if dataset != 'sbm': # real-world dataset
        print('------ Running {}.txt ------'.format(dataset))
        # read graph
        N, A = read_graph("datasets/{}.txt".format(dataset))
    else: # synthetic modified SBM
        pass
    A = (A + A.T)/2.0
    # compute signed laplacian
    Ap, An = (A>0), (A<0)
    clf = Cluster((Ap, An))
    if method == 'bnc-sym': C = clf.spectral_cluster_bnc(k=K, normalisation='sym')
    elif method == 'sponge': C = clf.SPONGE(k=K)
    elif method == 'sponge-sym': C = clf.SPONGE_sym(k=K)
    C = [x+1 for x in C]
    Y, obj = compute_polarity(C, A, K, N)
    return C, Y, A, N, K

opt = parse_arg()
if opt.K == None: raise Exception('Error: please specify K')
try: K = int(opt.K)
except ValueError: raise Exception('Error: please specify K in integer')

if opt.dataset == 'all':
    for dataset in DATASET_LIST:
        st = time.time()
        C, Y, A, N, _ = run(dataset, K+1, opt.method)
        ed = time.time()
        nums = [0 for i in range(K+1)]
        for x in C: nums[x-1]+=1
        idx = np.argsort(nums)[:K]
        groups = {}
        for x in idx: groups[x]=len(groups)
        C1 = []
        for x in C:
            if (x-1) in groups.keys(): C1+=[groups[x-1]+1]
            else: C1+=[-1]
        Y, obj = compute_polarity(C1, A, K, N)
        check_result_KCG(C1, Y, A, N, K, ed-st)
elif opt.dataset in DATASET_LIST:
    st = time.time()
    C, Y, A, N, K = run(opt.dataset, K, opt.method)
    ed = time.time()
    check_result_KCG(C, Y, A, N, K, ed-st)
elif opt.dataset == 'sbm': # synthetic graph from modified SBM
    try: N = int(opt.sbm_nv)
    except ValueError: raise Exception('Error: please specify the graph size in integer')
    try: nC = int(opt.sbm_nc)
    except ValueError: raise Exception('Error: please specify the community size in integer')
    for t in range(20):
        print('------------ [Round #{}] ------------'.format(t))
        for p in [0.1*(i) for i in range(10)]:
            print('------ Running SBM [p={:.1f}] ------'.format(p))
            _, A = gen_SBM(p, K, N, nC)
            st = time.time()
            C, Y, A, N, _ = run('sbm', K+1, opt.method, N=N, A=A)
            ed = time.time()
            precision, recall, precs, recs, f1_score, idx = compute_accuracy_Kplus1(C, nC, K)
            groups = {}
            for x in idx: groups[x]=len(groups)
            C1 = []
            for x in C:
                if (x-1) in groups.keys(): C1+=[groups[x-1]+1]
                else: C1+=[-1]
            Y, obj = compute_polarity(C1, A, K, N)
            check_result_KCG(C1, Y, A, N, K, ed-st)
            print('Accuracy: precision={:.2f}, recall={:.2f}, f1-score={:.2f}'.format(precision, recall, f1_score))
            print(precs)
            print(recs)
else:
    raise Exception('Error: please specify dataset name in {} or just leave it blank to run ALL'.format(DATASET_LIST))
