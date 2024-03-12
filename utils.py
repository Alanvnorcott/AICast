# utils.py

import numpy as np
import tensorflow as tf

def convert_state_to_tensor(current_state_dict):
    current_state_np = np.array([current_state_dict['population'], current_state_dict['resources'], current_state_dict['happiness']], dtype=np.float32)
    current_state_tensor = tf.convert_to_tensor(current_state_np, dtype=tf.float32)
    return current_state_tensor