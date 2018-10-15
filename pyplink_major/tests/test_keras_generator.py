from unittest import TestCase
import logging
import tensorflow as tf
from pyplink_major.plink_reader import Major_reader

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestMajor_reader(TestCase):

    def setUp(self):
        self.train = 'data/tf_keras_compare/sim_1000G_chr10_SampleMajor_train'
        self.dev = 'data/tf_keras_compare/sim_1000G_chr10_SampleMajor_dev'
        self.pheno = 'data/sim_1000G_chr10.txt'

    def test_generator(self):
        reader = Major_reader(self.train, self.pheno)
        dd = tf.data.Dataset()
        shape = (tf.TensorShape([59]), [])
        dataset = dd.from_generator(lambda: reader.one_iter('V1'),
                                    output_shapes=shape,
                                    output_types=(tf.float32, tf.float32))
        iterator = dataset.batch(10).repeat().make_one_shot_iterator()
        next_batch = iterator.get_next()

        with tf.Session() as sess:
            geno, pheno = sess.run(next_batch)
            self.assertEqual((10, 59), geno.shape)
            self.assertEqual((10, ), pheno.shape)

