############### SCG ###############
python SCG.py -k 2 -r min_angle > result/results_K2_SCG-MA.txt
python SCG.py -k 2 -r max_obj > result/results_K2_SCG-MO.txt
python SCG.py -k 2 -r randomized > result/results_K2_SCG-R.txt
python SCG.py -k 2 -r bansal > result/results_K2_SCG-B.txt
python SCG.py -k 6 -r min_angle > result/results_K6_SCG-MA.txt
python SCG.py -k 6 -r max_obj > result/results_K6_SCG-MO.txt
python SCG.py -k 6 -r randomized > result/results_K6_SCG-R.txt
python SCG.py -k 6 -r bansal > result/results_K6_SCG-B.txt
python SCG.py -k 6 -r min_angle -d sbm -n 2000 -c 100 > result/sbm_K6_SCG-MA.txt
python SCG.py -k 6 -r max_obj -d sbm -n 2000 -c 100 > result/sbm_K6_SCG-MO.txt
python SCG.py -k 6 -r randomized -d sbm -n 2000 -c 100 > result/sbm_K6_SCG-R.txt
python SCG.py -k 6 -r bansal -d sbm -n 2000 -c 100 > result/sbm_K6_SCG-B.txt

############### GroundTruth on m-SSBM ###############
python sbm_groundtruth.py -k 6 -n 2000 -c 100

############### KOCG(KDD'16) ###############
cd KOCG
python KOCG.py -f Run > ../result/KOCG_runtime_log.txt
python KOCG.py -f Eval -k 2 -t Top1
python KOCG.py -f Eval -k 6 -t Top1
python KOCG.py -f Eval -k 2 -t Topr
python KOCG.py -f Eval -k 6 -t Topr
python KOCG.py -d sbm -n 2000 -c 100 -f Run > ../result/sbm_K6_KOCG_runtime_log.txt
python KOCG.py -d sbm -n 2000 -c 100 -k 6 -t Top1 -f Eval
python KOCG.py -d sbm -n 2000 -c 100 -k 6 -t Topr -f Eval
cd ..

############### BNC(CIKM'12) ###############
python Baseline_SignedSpectralClustering_K.py -k 2 -m bnc-sym > result/results_K2_BNC_K.txt
python Baseline_SignedSpectralClustering_K.py -k 6 -m bnc-sym > result/results_K6_BNC_K.txt
python Baseline_SignedSpectralClustering_K.py -k 6 -m bnc-sym -d sbm -n 2000 -c 100 > result/sbm_K6_BNC_K.txt
python Baseline_SignedSpectralClustering_Kplus1.py -k 2 -m bnc-sym > result/results_K2_BNC_Kplus1.txt
python Baseline_SignedSpectralClustering_Kplus1.py -k 6 -m bnc-sym > result/results_K6_BNC_Kplus1.txt
python Baseline_SignedSpectralClustering_Kplus1.py -k 6 -m bnc-sym -d sbm -n 2000 -c 100 > result/sbm_K6_BNC_Kplus1.txt

############### SPONGE(AISTATS'19) ###############
python Baseline_SignedSpectralClustering_K.py -k 2 -m sponge > result/results_K2_SPONGE_K.txt
python Baseline_SignedSpectralClustering_K.py -k 6 -m sponge > result/results_K6_SPONGE_K.txt
python Baseline_SignedSpectralClustering_K.py -k 6 -m sponge-sym -d sbm -n 2000 -c 100 > result/sbm_K6_SPONGE_K.txt
python Baseline_SignedSpectralClustering_Kplus1.py -k 2 -m sponge > result/results_K2_SPONGE_Kplus1.txt
python Baseline_SignedSpectralClustering_Kplus1.py -k 6 -m sponge > result/results_K6_SPONGE_Kplus1.txt
python Baseline_SignedSpectralClustering_Kplus1.py -k 6 -m sponge-sym -d sbm -n 2000 -c 100 > result/sbm_K6_SPONGE_Kplus1.txt
