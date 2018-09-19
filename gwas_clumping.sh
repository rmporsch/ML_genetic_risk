#!/bin/sh

plink='./wepredict/bin/plink'
f='data/sim_1000G_chr10'
pheno='data/sim_1000G_chr10.txt'

##########
#  GWAS  #
##########
# $plink --bfile $f \
#   --pheno $pheno \
#   --pheno-name V1 \
#   --allow-no-sex \
#   --linear hide-covar \
#   --out 1kG

$plink --bfile $f \
  --clump 1kG.assoc.linear \
  --clump-p1 0.001 \
  --clump-p2 0.01 \
  --clump-r2 0.2 \
  --out 1kg
