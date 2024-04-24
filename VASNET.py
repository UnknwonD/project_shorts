import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, aperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()
        self.aperture = aperture
        self.ignore_itself = ignore_itself
        self.m = input_size
        self.output_size = output_size

        self.K = tf.keras.layers.Dense(output_size, use_bias=False)
        self.Q = tf.keras.layers.Dense(output_size, use_bias=False)
        self.V = tf.keras.layers.Dense(output_size, use_bias=False)
        self.output_linear = tf.keras.layers.Dense(input_size, use_bias=False)
        self.drop50 = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        n = tf.shape(x)[0]

        K = self.K(x)
        Q = self.Q(x) * 0.06
        V = self.V(x)

        logits = tf.matmul(Q, K, transpose_b=True)

        if self.ignore_itself:
            mask = 1 - tf.eye(n)
            logits = logits * mask - tf.float32.max * (1 - mask)

        if self.aperture > 0:
            mask = 1 - tf.linalg.band_part(tf.ones((n, n)), -self.aperture, self.aperture)
            logits = logits * mask - tf.float32.max * (1 - mask)

        att_weights = tf.nn.softmax(logits, axis=-1)
        weights = self.drop50(att_weights)
        y = tf.matmul(weights, V, transpose_a=True)
        y = self.output_linear(y)

        return y, att_weights

class VASNet(tf.keras.Model):
    def __init__(self):
        super(VASNet, self).__init__()
        self.m = 1024
        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = tf.keras.layers.Dense(1024)
        self.kb = tf.keras.layers.Dense(1024)
        self.kc = tf.keras.layers.Dense(1024)
        self.kd = tf.keras.layers.Dense(1)
        self.sig = tf.keras.activations.sigmoid
        self.relu = tf.keras.activations.relu
        self.drop50 = tf.keras.layers.Dropout(0.5)
        self.softmax = tf.keras.layers.Softmax(axis=0)
        self.layer_norm_y = tf.keras.layers.LayerNormalization()
        self.layer_norm_ka = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x, seq_len = inputs
        m = tf.shape(x)[2]
        x = tf.reshape(x, (-1, m))
        y, att_weights = self.att(x)

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = tf.reshape(y, (1, -1))

        return y, att_weights
