from tensorflow_example import *

def neuron_network(pfile, num_snps=100, learning_rate=0.01, epoch=1000, l=1.0, neuron_num = 10):
    genotypematrix = _get_matrix(pfile, num_snps)
    n, p = genotypematrix.shape
    pheno = np.random.normal(0, 1, n)  # random phenotype
    pheno = pheno.reshape((n, 1))

    # define placeholder for inputs to network
    x = tf.placeholder(tf.float32, shape=[None, p])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    # add hidden layer
    hidden_layer1 = add_layer(x, p, neuron_num, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(hidden_layer1, neuron_num, 1, activation_function=None)

    # the error between prediction and real data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), # sum the loss of all the samples
                         reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(epoch):
        # training
        sess.run(train_step, feed_dict={x: genotypematrix, y: pheno})
        if i % 50 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={x: genotypematrix, y: pheno}))

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

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

if  __name__ == '__main__':
    url = "ftp://climb.genomics.cn/pub/10.5524/100001_101000/100116/1kg_phase1_chr22.tar.gz"
    download_data(url, 'data')
    pfile = 'data/1kg_phase1_chr22'
    neuron_network(pfile)