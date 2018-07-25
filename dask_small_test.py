from dask.distributed import Client, LocalCluster
import dask
from wepredict.wepredict import wepredict

if __name__ == '__main__':
    plink_file = 'data/1kg_phase1_chr22'
    pheno_file = 'data/simulated_chr10.txt'
    ld_block_file = 'data/Berisa.EUR.hg19.bed'
    monster = wepredict(plink_file, ld_block_file,
                            testing=True)


    cluster = LocalCluster(n_workers=3)
    client = Client(cluster)
    pheno = monster.simulate()
    out = dask.compute(pheno)
    cluster.close()
    print('done')