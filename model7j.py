import os
import tensorflow as tf

GRU_UNITS = 1024
OUTPUT_SIZE = 300


class Model(tf.keras.Model):
    def __init__(self, units=GRU_UNITS, output_size=OUTPUT_SIZE,
                 checkpoint_dir='./model7j_training_checkpoints'):
        super(Model, self).__init__()
        self.units = units
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                    self.units,
                    return_sequences=True,
                    return_state=True,
                    recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(
                    self.units,
                    return_sequences=True,
                    return_state=True,
                    recurrent_activation='sigmoid',
                    recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(OUTPUT_SIZE)
        self.optimizer = tf.train.AdamOptimizer()
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                                  model=self)

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(
            self.checkpoint_dir))

    def call(self, x, hidden):
        output, states = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, states
