source myenv_lc/bin/activate
parallel -j 16 :::: params_Qsigma_LC.txt
