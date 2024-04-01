#!/bin/bash
plink --tfile unitigs --make-bed
./gcta64/gcta64  --bfile plink  --simu-qt --simu-causal-loci causal10.snplist  --simu-hsq 1 --simu-rep 1   --out out10
./gcta64/gcta64  --bfile plink  --simu-qt --simu-causal-loci causal50.snplist  --simu-hsq 1 --simu-rep 1   --out out50
./gcta64/gcta64  --bfile plink  --simu-qt --simu-causal-loci causal200.snplist  --simu-hsq 1 --simu-rep 1   --out out200
cp out* ./pheno_data/