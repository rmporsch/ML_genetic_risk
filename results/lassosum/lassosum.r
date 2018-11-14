#!/usr/bin/env Rscript

library(lassosum)
library(data.table)
args = commandArgs(trailingOnly=TRUE)
setwd("/home/rmporsch/projects/ML_genetic_risk/results")

ref = args[1]
dev = args[2]
sumstat = args[3]
ld = args[4]
n = args[5]
phenotype = args[6]
pheno.name = args[7]

ld  <- fread(ld)
ss = fread(sumstat)
# ss$P = as.numeric(ss$P, digits=128)
ss$P[is.infinite(ss$P)] = 1e-128
print(min(ss$P))
cor = p2cor(p = as.numeric(ss$P), n=as.integer(n), sign=as.numeric(ss$BETA))
cor[is.na(cor)] = 0.999999


out <- lassosum.pipeline(cor=cor, chr=ss$CHR, pos=ss$BP, 
                         A1=ss$A1,
                         ref.bfile=ref, test.bfile=dev, 
                         LDblocks = ld)

dat = read.table(phenotype, head=T)
dev_fam = read.table(paste0(out$test.bfile, '.fam'))
names(dev_fam)[1:2] = c('FID', 'IID')
sub = merge(dev_fam, dat, by=c('FID', 'IID'))
ssub = sub[,c('FID', 'IID', pheno.name)]
ssub = dat[,c('FID', 'IID', pheno.name)]
v <- validate(out, pheno=ssub)
output = v$validation.table

output.name = paste(dev,'phenotype', 'lasso', sep='.') 
write.table(output, output.name)
