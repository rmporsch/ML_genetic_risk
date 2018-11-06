#!/bin/bash
script='/home/rmporsch/projects/ML_genetic_risk/bin/process.py'

plinkfile='/home/tshmak/DATA/UKBB/data/maf_0.01_10'
outputfolder='data/sample_major/ukb/clumped/'
dev='data/sample_major/ukb/maf_0.01_10_SampleMajor_dev.fam'
train='data/sample_major/ukb/maf_0.01_10_SampleMajor_train.fam'
pheno_file='data/simulated_chr10.txt'
phenotype='V1'
# plink='spyplink/bin/plink'
# plink2='spyplink/bin/plink2'

python $script \
  $plinkfile \
  $outputfolder \
  $train $dev \
  $pheno_file \
  $phenotype \
  -p1=0.0001 \
  -p2=0.01 \
  -r2=0.5 -d
