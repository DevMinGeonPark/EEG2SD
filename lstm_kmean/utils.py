# Description: Utility functions for the project
import numpy as np
import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
from functools import partial

data_cls = natsorted(glob('data/thoughtviz_eeg_data/*'))
cls2idx  = {key.split(os.path.sep)[-1]:idx for idx, key in enumerate(data_cls, start=0)}
idx2cls  = {value:key for key, value in cls2idx.items()}

def preprocess_data(X, Y):
	X = tf.squeeze(X, axis=-1)
	max_val = tf.reduce_max(X)/2.0
	X = (X - max_val) / max_val
	X = tf.transpose(X, [1, 0])
	X = tf.cast(X, dtype=tf.float32)
	Y = tf.argmax(Y)
	return X, Y

def load_complete_data(X, Y, batch_size=16):	
	dataset = tf.data.Dataset.from_tensor_slices((X, Y)).map(preprocess_data).shuffle(buffer_size=2*batch_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
	return dataset