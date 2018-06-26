"""Convert a plink bed file into LD blocks."""
from wepredict.plink_reader import Genetic_data_read
from wepredict.data_download import genetic_testdata
import sys

plink_step = sys.argv[1]
chr = sys.argv[2]
download_path = sys.argv[3]
print(sys.argv)

downloader = genetic_testdata(download_path)
ld_blocks = downloader.download_ldblocks()

genetic_process = Genetic_data_read(plink_step, ld_blocks)
out = genetic_process.block_iter(int(chr))
for i in out:
    x = i
