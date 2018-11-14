#!/bin/bash
#PBS -l nodes=1:ppn=10 
#PBS -l mem=30gb 
#PBS -l walltime=24:00:00 
#PBS -q psychipc
#PBS -N convering_UKB_sample_major

cd /home/rmporsch/projects/ML_genetic_risk

script='/home/rmporsch/projects/ML_genetic_risk/bin/process.py'

plinkfile='/home/tshmak/DATA/UKBB/data/maf_0.01_10'
outputfolder='data/sample_major/ukb/clumped/nonlinear/'
dev='data/sample_major/ukb/maf_0.01_10_SampleMajor_dev.fam'
train='data/sample_major/ukb/maf_0.01_10_SampleMajor_train.fam'
pheno_file='data/simulated_chr10.txt'
pheno_file='data/pseudophenos_mini.txt'
phenotype='V1'

python $script \
  $plinkfile \
  $outputfolder \
  $train $dev \
  $pheno_file \
  $phenotype \
  -p1=1.0 \
  -p2=1.0 \
  -r2=0.2 -d
