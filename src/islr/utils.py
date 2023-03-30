import os
import random
import numpy as np
import tensorflow as tf

def seed_it_all(seed=42):
    """
    Seeds all random number generators used in the project to ensure reproducibility.
    
    Args:
    - seed (int): The random seed to use. Default is 42.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def print_shape_dtype(var_list, names):
    """
    Prints shape and dtype for list of variables

    Args:
    - var_list (list): List of numpy variables
    - names (str): Name of the variable to print
    """
    for e, n in zip(var_list, names):
        print(f'{n} shape: {e.shape}, dtype: {e.dtype}')