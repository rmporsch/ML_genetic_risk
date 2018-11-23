from setuptools import setup

setup(name='wepredict',
      version='1.0',
      description='LD-Block prediction',
      author='Robert M. Porsch',
      author_email='rmporsch@gmail.com',
      install_requires=["bitarray", "matplotlib", "numpy", "pandas", "scipy", "setuptools", "dask", "glob2", "protobuf", "pyplink", "scikit_learn", "tensorflow", "torch", "typing"],
      packages=['wepredict', 'nnpredict', 'spyplink'],
      test_suit='tests'
     )
