import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle as pkl
import pandas as pd
import optparse
import os

EPS = 1E-10
matplotlib.rcParams.update({'font.size': 16})
datasets = ['wow8', 'bitcoin', 'wikivot', 'referendum', 'slashdot', 'wikicon', 'epinions','wikipol']
x = np.arange(len(datasets))
width = 0.6  # the width of the bars
fsize = (15,5)

def parse_arg():
    """ utility function to parse argument from command line """
    parser = optparse.OptionParser()
    parser.add_option('-o', dest='output_dir', default='figs', help='specify the folder name to put plots')
    (options, args) = parser.parse_args()
    return options

opt = parse_arg()
if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)
################################ Real-World Experiment ################################
for k in [2,6]:
    print('############### [K={}] ###############'.format(k))
    for method,fname in zip(
        ['SCG-MA','SCG-MO','SCG-B','SCG-R','KOCG-Top1','KOCG-Topr','BNC-k','BNC-(k+1)','SPONGE-k','SPONGE-(k+1)'],
        ['results_K{}_SCG-MA.txt','results_K{}_SCG-MO.txt','results_K{}_SCG-B.txt','results_K{}_SCG-R.txt',
        'results_K{}_KOCG_Top1.txt','results_K{}_KOCG_Topr.txt','results_K{}_BNC_K.txt','results_K{}_BNC_Kplus1.txt',
        'results_K{}_SPONGE_K.txt','results_K{}_SPONGE_Kplus1.txt']):
        # K-PC[deterministic rounding: minimum angle]
        with open(fname.format(k)) as f:
            data = f.read()
        if 'KOCG' not in method:
            polarity = {d:float(x) for d,x in zip(re.findall('------ Running (.*).txt ------\n', data), re.findall('Obj = (.*) in .*\n', data))}
            exe_time = {d:float(x) for d,x in zip(re.findall('------ Running (.*).txt ------\n', data), re.findall('Obj = .*, Execution Time=(.*)\n', data))}
        else:
            polarity = {d:float(x) for d,x in zip(re.findall('------ Running (.*).txt ------\n', data), re.findall('Obj = (.*)\n', data))}
            with open('KOCG_runtime_log.txt'.format(k)) as f:
                data = f.read()
            if k==2: t = re.findall('  enumKOCG Complete! Time: (\d*.\d*) =', data)[:8]
            elif k==6: t = re.findall('  enumKOCG Complete! Time: (\d*.\d*) =', data)[8:]
            exe_time = {d:float(x.split(' ')[0]) for d,x in zip(re.findall(' Running (\w*) =', data),t)}
        print('[{}]'.format(method))
        for d in datasets:
            if d in polarity.keys():
                print('\t{}:\tpolarity={:.1f}\ttime={:.1f}'.format(d, polarity[d], exe_time[d]))

# plot group size
Sizes = {}
fnames = ['results_K6_SCG-MA.txt', 'results_K6_SCG-MO.txt', 'results_K6_SCG-B.txt', 'results_K6_SCG-R.txt',
        'results_K6_KOCG_Top1.txt', 'results_K6_KOCG_Topr.txt', 'results_K6_BNC_K.txt', 'results_K6_BNC_Kplus1.txt',
        'results_K6_SPONGE_K.txt', 'results_K6_SPONGE_Kplus1.txt']
dnames = ['SCG-MA', 'SCG-MO', 'SCG-B', 'SCG-R', 'KOCG-Top-1', 'KOCG-Top-r', 'BNC-k', 'BNC-(k+1)', 'SPONGE-k', 'SPONGE-(k+1)']
for fname,dname in zip(fnames, dnames):
    with open(fname, 'r') as f:
        data = f.readlines()
    Gs, vs = {}, []
    for l in data:
        if 'Running' in l:
            d = re.findall('------ Running (.*).txt ------', l)[0]
            if d not in Gs: Gs[d] = []
        elif '|In_+|' in l and 'Total' not in l:
            vs += [int(re.findall('\|S\_.*\|=(.*), \|In', l)[0])]
        elif 'neutral' in l:
            vs = sorted(vs, reverse=True)
            for j in range(len(vs),6): vs += [0]
            Gs[d] += [vs.copy()]
            vs = []
    Sizes[dname] = Gs.copy()
DName = {d:x for d,x in zip(datasets,['WoW-EP8', 'Bitcoin', 'WikiVot', 'Referendum', 'Slashdot', 'WikiCon', 'Epinions', 'Wikipol'])}
for dataset in DName.keys():
    X = {}
    for x in dnames:
        if dataset in Sizes[x]:X[x] = Sizes[x][dataset][0]
    df = pd.DataFrame(X)
    plt.figure(figsize=(20,3))
    ax = boxplot = df.boxplot(return_type='axes', grid=False)
    ax.set_yscale('log')
    ax.set_ylabel('$|S_t|$')
    plt.title('[{}] Detected Group Size'.format(DName[dataset]))
    plt.savefig('{}/compare_groupsize_{}.pdf'.format(opt.output_dir, DName[dataset]))

################################ Modified SBM Experiment ################################
F1Scores, F1Std, Polarity, PolarityStd,DisRatio, DisRatioStd = {},{},{},{},{},{}
for name in ['GroundTruth','SCG-MA', 'SCG-MO', 'SCG-B', 'SCG-R', 'KOCG_Topr', 'KOCG_Top1', 'BNC_K', 'BNC_Kplus1', 'SPONGE_K', 'SPONGE_Kplus1']:
    with open('sbm_K6_{}.txt'.format(name)) as f:
        data = f.readlines()
    x1,x2,x3,s1,s2,s3,aa = {},{},{},{},{},{},[]
    for l in data:
        if 'Running SBM' in l:
            p = re.findall('------ Running SBM \[p=(.*)\] ------', l)[0]
            if p not in x1.keys(): x1[p],x2[p],x3[p],s1[p],s2[p],s3[p] = [],[],[],0,0,0
        elif '|In_+|' in l and 'Total' not in l:
            a1, a2 = 0, 0
            s = re.findall('\|In\_\+\|-\|In\_-\|=(.*), \|Out', l)[0].split('-')
            a1 += int(s[0])
            a2 += int(s[1])
            s = re.findall('\|Out\_-\|-\|Out\_\+\|=(.*)', l)[0].split('-')
            a1 += int(s[0])
            a2 += int(s[1])
            aa += [a1/(a1+a2+EPS)]
        elif 'Obj = ' in l:
            if ('KOCG' not in name)&('GroundTruth' not in name):
                t = float(re.findall('Obj = (.*) in .*\n', l)[0])
            else:
                t = float(re.findall('Obj = (.*)\n', l)[0])
            x2[p] += [t]
        elif 'Accuracy: ' in l:
            x1[p] += [float(re.findall('Accuracy: .* f1-score=(.*)\n', l)[0])]
            x3[p] += [np.mean([1-i for i in aa])]
            aa = []
    for i in range(7):
        p = '{:.1f}'.format(i*0.1)
        s1[p],s2[p],s3[p] = np.std(x1[p]),np.std(x2[p]),np.std(x3[p])
        x1[p],x2[p],x3[p] = np.mean(x1[p]),np.mean(x2[p]),np.mean(x3[p])
    F1Scores[name] = x1.copy()
    Polarity[name] = x2.copy()
    DisRatio[name] = x3.copy()
    F1Std[name] = s1.copy()
    PolarityStd[name] = s2.copy()
    DisRatioStd[name] = s3.copy()
probs = ['{:.1f}'.format(i*0.1) for i in range(7)]
# f1-score vs edge density
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
x = np.arange(7)
w = 5E-3
axs[0].errorbar(x-5*w, [F1Scores['SCG-MA'][p] for p in probs], yerr=[F1Std['SCG-MA'][p] for p in probs], marker='o', label='SCG-MA', linewidth=2, color='C1')
axs[0].errorbar(x-4*w, [F1Scores['SCG-MO'][p] for p in probs], yerr=[F1Std['SCG-MO'][p] for p in probs], marker='o', label='SCG-MO', linewidth=2, color='C2')
axs[0].errorbar(x-3*w, [F1Scores['SCG-B'][p] for p in probs], yerr=[F1Std['SCG-B'][p] for p in probs], marker='o', label='SCG-B', linewidth=2, color='C3')
axs[0].errorbar(x-2*w, [F1Scores['SCG-R'][p] for p in probs], yerr=[F1Std['SCG-R'][p] for p in probs], marker='o', label='SCG-R', linewidth=2, color='C4')
axs[0].errorbar(x-w, [F1Scores['KOCG_Top1'][p] for p in probs], yerr=[F1Std['KOCG_Top1'][p] for p in probs], marker='x', label='KOCG-top-1', linewidth=2, color='C5')
axs[0].errorbar(x, [F1Scores['KOCG_Topr'][p] for p in probs], yerr=[F1Std['KOCG_Topr'][p] for p in probs], marker='x', label='KOCG-top-r', linewidth=2, color='C6')
axs[0].errorbar(x+w, [F1Scores['BNC_K'][p] for p in probs], yerr=[F1Std['BNC_K'][p] for p in probs], marker='*', label='BNC-k', linewidth=2, color='C7')
axs[0].errorbar(x+2*w, [F1Scores['BNC_Kplus1'][p] for p in probs], yerr=[F1Std['BNC_Kplus1'][p] for p in probs], marker='*', label='BNC-(k+1)', linewidth=2, color='C10')
axs[0].errorbar(x+3*w, [F1Scores['SPONGE_Kplus1'][p] for p in probs], yerr=[F1Std['SPONGE_Kplus1'][p] for p in probs], marker='+', label='SPONGE-(k+1)', linewidth=2, color='black')
axs[0].errorbar(x+4*w, [F1Scores['SPONGE_K'][p] for p in probs], yerr=[F1Std['SPONGE_K'][p] for p in probs], marker='+', label='SPONGE-k', linewidth=2, color='C8')
axs[0].errorbar(x+5*w, [F1Scores['GroundTruth'][p] for p in probs], yerr=[F1Std['GroundTruth'][p] for p in probs], ls='-.', label='GroundTruth', linewidth=1, color='C9')
axs[0].set_ylabel('F1-Score')
axs[0].set_xlabel('$\eta$')
axs[0].set_xticks(x)
axs[0].set_xticklabels(probs)
axs[0].set_title('(a) F1-Score vs $\eta$')
# polarity vs edge density
axs[1].errorbar(x-5*w, [Polarity['SCG-MA'][p] for p in probs], yerr=[PolarityStd['SCG-MA'][p] for p in probs], marker='o', label='SCG-MA', linewidth=2, color='C1')
axs[1].errorbar(x-4*w, [Polarity['SCG-MO'][p] for p in probs], yerr=[PolarityStd['SCG-MO'][p] for p in probs], marker='o', label='SCG-MO', linewidth=2, color='C2')
axs[1].errorbar(x-3*w, [Polarity['SCG-B'][p] for p in probs], yerr=[PolarityStd['SCG-B'][p] for p in probs], marker='o', label='SCG-B', linewidth=2, color='C3')
axs[1].errorbar(x-2*w, [Polarity['SCG-R'][p] for p in probs], yerr=[PolarityStd['SCG-R'][p] for p in probs], marker='o', label='SCG-R', linewidth=2, color='C4')
axs[1].errorbar(x-w, [Polarity['KOCG_Top1'][p] for p in probs], yerr=[PolarityStd['KOCG_Top1'][p] for p in probs], marker='x', label='KOCG-top-1', linewidth=2, color='C5')
axs[1].errorbar(x, [Polarity['KOCG_Topr'][p] for p in probs], yerr=[PolarityStd['KOCG_Topr'][p] for p in probs], marker='x', label='KOCG-top-r', linewidth=2, color='C6')
axs[1].errorbar(x+w, [Polarity['BNC_K'][p] for p in probs], yerr=[PolarityStd['BNC_K'][p] for p in probs], marker='*', label='BNC-k', linewidth=2, color='C7')
axs[1].errorbar(x+2*w, [Polarity['BNC_Kplus1'][p] for p in probs], yerr=[PolarityStd['BNC_Kplus1'][p] for p in probs], marker='*', label='BNC-(k+1)', linewidth=2, color='C10')
axs[1].errorbar(x+3*w, [Polarity['SPONGE_Kplus1'][p] for p in probs], yerr=[PolarityStd['SPONGE_Kplus1'][p] for p in probs], marker='+', label='SPONGE-(k+1)', linewidth=2, color='black')
axs[1].errorbar(x+4*w, [Polarity['SPONGE_K'][p] for p in probs], yerr=[PolarityStd['SPONGE_K'][p] for p in probs], marker='+', label='SPONGE-k', linewidth=2, color='C8')
axs[1].errorbar(x+5*w, [Polarity['GroundTruth'][p] for p in probs], yerr=[PolarityStd['GroundTruth'][p] for p in probs], ls='-.', label='GroundTruth', linewidth=1, color='C9')
axs[1].set_ylabel('Polarity')
axs[1].set_xlabel('$\eta$')
axs[1].set_xticks(x)
axs[1].set_xticklabels(probs)
axs[1].set_title('(b) Polarity vs $\eta$')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('{}/compare_sbm_with_errorbar.pdf'.format(opt.output_dir))
