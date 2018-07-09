"""Simple linear regression with different norms."""
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from tensorflow_example import download_data
from pytorch_regression import pytorch_linear
import os
import pandas as pd
import numpy as np
from typing import Tuple, Any


def get_test_data(d_path: str = 'tensor/data') -> Tuple[Any, Any]:
    """Download and processing of test data."""
    DATA_PATH = os.path.abspath(d_path)
    TESTDATA_URL = "https://archive.ics.uci.edu/ml/" \
        "machine-learning-databases/undocumented/" \
        "connectionist-bench/sonar/sonar.all-data"
    download_data(TESTDATA_URL, DATA_PATH)

    testdata = pd.read_csv(os.path.join(DATA_PATH, 'sonar.all-data'),
                           header=None)
    print("Test data has ", testdata.shape[0], "rows")
    print("Test data has ", testdata.shape[1], "features")
    X = scale(testdata.iloc[:, :-1])

    y = testdata.iloc[:, -1].values
    encoder = LabelEncoder()
    encoder.fit(np.unique(y))
    y = encoder.transform(y)
    return X, y

    
if __name__ == '__main__':
    DATA_PATH = 'tensor/data'
    model_comparision_file = os.path.join(DATA_PATH, 'model.comparisions')
    X, y = get_test_data(DATA_PATH)

    pytorchmodel = pytorch_linear(X, y, model_comparision_file, True)
    pytorchmodel.run(penal='l0')
