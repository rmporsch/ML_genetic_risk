"""Convert a plink bed file into LD blocks."""
from wepredict.plink_reader import Genetic_data_read
from wepredict.data_download import genetic_testdata
import sys

download_path = sys.arg[2]
plink_step = sys.arg[1]

downloader = genetic_testdata(download_path)
ld_blocks = downloader.download_ldblocks()

genetic_process = Genetic_data_read(plink_step, ld_blocks)
out = genetic_process.block_iter(10)
for i in out:
    x = i
