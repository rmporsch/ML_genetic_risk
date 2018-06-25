import numpy as np
import tensorflow as tf
from pyplink import PyPlink
import sys
import os
import urllib.request
import tarfile
import zipfile

print("Tensorflow version : {}".format(tf.VERSION))


def _get_matrix(pfile, max_block):
    """Extract a genotype matrix from plink file."""
    with PyPlink(pfile) as bed:
        bim = bed.get_bim()
        fam = bed.get_fam()
        n = fam.shape[0]
        p = bim.shape[0]
        assert(max_block <= p)
        genotypemat = np.zeros((n, max_block), dtype=np.int64)
        u = 0
        for loci_name, genotypes in bed:
            genotypemat[:, u] = np.array(genotypes)
            u += 1
            if u >= max_block:
                break
        return genotypemat


def _print_download_progress(count, block_size, total_size):
    """Use for monitoring downloading process."""
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def download_data(url, download_dir):
    """Download data from url."""
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


def linear_regression(pfile, num_snps=100, learning_rate=0.01, epoch=100, l=1.0):
    """Run a Simple linear regression with tensorflow."""
    # data
    genotypematrix = _get_matrix(pfile, num_snps)
    n, p = genotypematrix.shape
    pheno = np.random.normal(0, 1, n)  # random phenotype
    pheno = pheno.reshape((n, 1))
    x = tf.placeholder(tf.float32, shape=[n, p])
    y = tf.placeholder(tf.float32, shape=[n, 1])
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat().batch(100)

    # model
    W = tf.Variable(tf.ones([p, 1]))
    init = tf.global_variables_initializer()
    y_ = tf.matmul(x, W)
    cost = tf.reduce_mean(tf.square(y_-y)) + l*tf.reduce_sum(tf.abs(W))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    feed_dict = {x: genotypematrix, y: pheno}

    cost_history = np.empty([0], dtype=float)
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(epoch):
            sess.run(train_op, feed_dict=feed_dict)
            current_cost = sess.run(cost, feed_dict=feed_dict)
            print("loss = %f" % np.round(current_cost, 5))
            cost_history = np.append(cost_history, current_cost)

    return cost_history


if __name__ == '__main__':
    url = "ftp://climb.genomics.cn/pub/10.5524/100001_101000/100116/1kg_phase1_chr22.tar.gz"
    download_data(url, 'data')
    pfile = 'data/1kg_phase1_chr22'
    linear_regression(pfile)
