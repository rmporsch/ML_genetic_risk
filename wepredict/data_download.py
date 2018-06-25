"""Download data from google cloud storage."""
from google.cloud import storage
import os


class genetic_testdata(object):
    """Import and process genetic data in plink format for ML."""

    def __init__(self, download_path: str):
        """Perpare genetic data."""
        super(genetic_testdata, self).__init__()
        self.download_path = download_path
        self._gc_client = storage.Client('Hail')
        self._bucket = self._gc_client.get_bucket('ukb_testdata')

    def _download(self, gc_path: str, output_path: str):
        if ('gs' in gc_path) or ('ukb_testdata' in gc_path):
            raise ValueError('path is the path AFTER the bucket name')
        blob = self._bucket.blob(gc_path)
        blob.download_to_filename(output_path)

    def download_1kg_chr22(self) -> str:
        """Download chr22 from the 1K Genome."""
        files = ['1kg_phase1_chr22.' + k for k in ['bed', 'bim', 'fam']]
        print('start downloading files')
        for f in files:
            output_path = os.path.join(self.download_path, f)
            gc_path = os.path.join('data', f)
            if os.path.isfile(output_path):
                continue
            self._download(gc_path, output_path)
        print('Files were donwloaded to {}'.format(self.download_path))
        return os.path.join(self.download_path, '1kg_phase1_chr22')

    def download_ukb_chr10(self) -> str:
        """Download chr10 from the UKB (maf>=0.01)."""
        files = ['maf_0.01_10.' + k for k in ['bed', 'bim', 'fam']]
        print('start downloading files')
        for f in files:
            output_path = os.path.join(self.download_path, f)
            gc_path = os.path.join('data', f)
            if os.path.isfile(output_path):
                continue
            self._download(gc_path, output_path)
        print('Files were donwloaded to {}'.format(self.download_path))
        return os.path.join(self.download_path, 'maf_0.01_10')

    def download_ldblocks(self) -> str:
        """Download LD block file."""
        file = 'Berisa.EUR.hg19.bed'
        output_path = os.path.join(self.download_path, file)
        gc_path = os.path.join('data', file)
        if not os.path.isfile(output_path):
            self._download(gc_path, output_path)
        return os.path.join(self.download_path, file)

    def download_file(self, file: str) -> str:
        """Download any given file."""
        output_path = os.path.join(self.download_path, os.path.basename(file))
        gc_path = file
        if not os.path.isfile(output_path):
            self._download(gc_path, output_path)
        return output_path
