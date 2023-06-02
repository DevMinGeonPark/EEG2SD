import tensorflow as tf
from tensorflow.keras import Model, models, layers, regularizers
import tensorflow_addons as tfa

weight_decay = 1e-4

def enc_conv_block(filters, kernel, strides, padding, rate):
	return models.Sequential([
			layers.Conv1D(filters=filters, kernel_size=kernel, strides=strides, padding=padding),
			layers.Activation(activation='leaky_relu'),
			layers.BatchNormalization(),
			layers.Dropout(rate=rate)
		])


class TripleNet(Model):
	def __init__(self, n_classes=10, n_features=128):
		super(TripleNet, self).__init__()
		filters   = [ 32,  n_features]
		ret_seq   = [ True, False]
		self.enc_depth  = len(filters)
		self.encoder   = [layers.LSTM(units=filters[idx], return_sequences=ret_seq[idx]) for idx in range(self.enc_depth)]
		self.flat      = layers.Flatten()
		self.w_1       = layers.Dense(units=n_features, activation='leaky_relu')
		self.w_2       = layers.Dense(units=n_features)

	def call(self, x):
		for idx in range(self.enc_depth):
			x = self.encoder[idx]( x )
		x = feat = self.flat( x )
		x = tf.nn.l2_normalize(x, axis=-1)

		return x, feat

@tf.function
def train_step(softnet, opt, X, Y):
	with tf.GradientTape() as tape:
		Y_emb, _ = softnet(X, training=True)
		loss  = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)
	variables = softnet.trainable_variables
	gradients = tape.gradient(loss, variables)
	opt.apply_gradients(zip(gradients, variables))
	return loss

@tf.function
def test_step(softnet, X, Y):
	Y_emb, _ = softnet(X, training=False)
	loss  = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)
	return loss