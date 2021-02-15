# SCG: Discovering Conflicting Groups in Signed Networks (NeurIPS2020)
"Discovering Conflicting Groups in Signed Networks", Ruo-Chun Tzeng, Bruno Ordozgoiti, and Aristides Gionis, In Proc. of NeurIPS 2020.
 * [paper](https://proceedings.neurips.cc//paper_files/paper/2020/hash/7cc538b1337957dae283c30ad46def38-Abstract.html), [video](https://youtu.be/xDCjBeS5uHc).

## 1. Dependency

### 1.1. Ours SCG Methods
 * Python 3.7
 * NumPy 1.17
 * SciPy 1.3

### 1.2. Baselines
 * KOCG(KDD'16): install `matlab_engine` via the link https://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html.
 * BNC/SPONGE: `pip install git+https://github.com/alan-turing-institute/SigNet.git`.

## 2. To Reproduce Experimental Results

### 2.1. Run-Experiments
All the required commands are in the bash file `run.sh`, you just need to run it.
 * `chmod 755 run.sh` to give execution permission.
 * `./run.sh`.

### 2.2. Inspect-Results
 * Go to `result/` folder.
 * Run `python plot.py -o figs_reproduce` will print out summarized result and plot figures to the specified folder, ex: `figs_reproduce/`

## 3. Note
All methods except SCG-R and KOCG should be exactly identical to our reported figures on real-world datasets in the paper.
 * The reason why re-running the baseline KOCG(KDD'16) might have slightly different result is because of its random initialization (roulette wheel selection). However, it is certain that their method results in much lower polarity scores than our SCG methods.
 * For the baseline SPONGE(AISTATS'19), we try both `unnormalized` and `symmetric normalized` normalization scheme in their SigNet implementation and report the best of the two in both real-world dataseta and synthetic m-SSBM networks.
