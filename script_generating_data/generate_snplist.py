#!/usr/bin/env python
# coding : utf-8

import os
ns = [10,50,200]
count = 0
fw = open('causal.snplist','w',encoding='utf-8')
with open('./originaldata/unitigs_trim.Rtab','r') as f:
    lines = f.readlines()
    for index , line in enumerate(lines):
        count += 1
        fw.write('unitg'+str(index)+'\n')
fw.close()


for n in ns:
    dns = count//n 
    file = open('causal'+str(n)+'.snplist','w',encoding='utf-8')
    for ii in range(0,n):
        file.write('unitig'+str(ii*dns)+'\n')
    file.close() 

file = open('gwas_simu.sh','w',encoding='utf-8')
file.write('#!/bin/bash\n')
file.write('plink --tfile unitigs --make-bed\n')
for n in ns:
    file.write('./gcta64/gcta64  --bfile plink  --simu-qt --simu-causal-loci causal'+str(n)+'.snplist  --simu-hsq 1 --simu-rep 1   --out out'+str(n)+'\n')
  
file.write('cp out* ./pheno_data/')
file.close()
