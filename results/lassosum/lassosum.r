#!/usr/bin/env Rscript

library(lassosum)
library(data.table)
args = commandArgs(trailingOnly=TRUE)

ref = args[1]
dev = args[2]
sumstat = args[3]
ld = args[4]
n = args[5]
phenotype = args[6]

ld  <- fread(ld)
ss = fread(sumstat)
cor = p2cor(p = ss$P, n=n, sign=ss$BETA)

out <- lassosum.pipeline(cor=cor, chr=ss$CHR, pos=ss$BP, 
                         A1=ss$A1,
                         ref.bfile=ref, test.bfile=dev, 
                         LDblocks = ld)

dat = read.table(phenotype, head=T)
dev_fam = read.table(paste0(out$test.bfile, '.fam'))
names(dev_fam)[1:2] = c('FID', 'IID')
sub = merge(dev_fam, dat, by=c('FID', 'IID'))
ssub = sub[,c('FID', 'IID', 'V1')]
ssub = dat[,c('FID', 'IID', 'V1')]
v <- validate(out, pheno=ssub)
output = v$validation.table

output.name = paste(dev,'phenotype', 'lasso', sep='.') 
write.table(output, output.name)
