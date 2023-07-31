import numpy as np
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
from settings import CLASSIFICATION_KEY, REGRESSION_KEY

def build_data_interface(train_loader, test_loader, homogeneous):
    return Data(train_loader, test_loader, homogeneous)

class Data:

    def __init__(self, train_loader, test_loader, homogeneous):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.homogeneous = homogeneous