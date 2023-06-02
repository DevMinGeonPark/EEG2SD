# Description: Training script for the LSTM-KMeans model
import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pickle
from model import TripleNet, train_step, test_step
from utils import load_complete_data
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

style.use('seaborn')

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

np.random.seed(45)
tf.random.set_seed(45)

if __name__ == '__main__':
	n_channels  = 14
	n_feat      = 128
	batch_size  = 256
	test_batch_size  = 1
	n_classes   = 10

	with open('data/eeg/image/data.pkl', 'rb') as file:
		data = pickle.load(file, encoding='latin1')
		train_X = data['x_train']
		train_Y = data['y_train']
		test_X = data['x_test']
		test_Y = data['y_test']

	train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
	val_batch   = load_complete_data(test_X, test_Y, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
	X, Y = next(iter(train_batch))

	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='experiments/best_ckpt', max_to_keep=5000)
	triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
	START = int(triplenet_ckpt.step) // len(train_batch)
	if triplenet_ckptman.latest_checkpoint:
		print('Restored from the latest checkpoint, epoch: {}'.format(START))
	EPOCHS = 3000
	cfreq  = 178 

	kmeanacc = 0.0
	tq = tqdm(test_batch)
	feat_X = []
	feat_Y = []
	for idx, (X, Y) in enumerate(tq, start=1):
		_, feat = triplenet(X, training=False)
		feat_X.extend(feat.numpy())
		feat_Y.extend(Y.numpy())
	feat_X = np.array(feat_X)
	feat_Y = np.array(feat_Y)
	print(feat_X.shape, feat_Y.shape)

	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
	tsne_results = tsne.fit_transform(feat_X)
	df = pd.DataFrame()
	df['label'] = feat_Y
	df['x1'] = tsne_results[:, 0]
	df['x2'] = tsne_results[:, 1]
 
	if not os.path.exists('./experiments/inference'): os.makedirs('./experiments/inference')

	df.to_csv('experiments/inference/triplet_embed2D.csv')
	
	df = pd.read_csv('experiments/inference/triplet_embed2D.csv')

	plt.figure(figsize=(16,10))
	
	sns.scatterplot(
	    x="x1", y="x2",
	    data=df,
	    hue='label',
	    palette=sns.color_palette("hls", n_classes),
	    legend="full",
	    alpha=0.4
	    )

	plt.show()

	kmeans = KMeans(n_clusters=n_classes,random_state=45)
	kmeans.fit(feat_X)
	labels = kmeans.labels_

	correct_labels = sum(feat_Y == labels)
	print("Result: %d out of %d samples were correctly labeled." % (correct_labels, feat_Y.shape[0]))
	kmeanacc = correct_labels/float(feat_Y.shape[0])
	print('Accuracy score: {0:0.2f}'. format(kmeanacc))